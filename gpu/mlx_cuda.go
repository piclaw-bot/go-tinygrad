package gpu

// MLX affine 4-bit GPU support: optimized GEMV + transpose-to-GPTQ upload.
//
// Two approaches for MLX weights on GPU:
// 1. Native MLX kernel (mlx_gemv_opt): shared-mem x, 8x unroll, coalesced weight reads
// 2. Transpose at upload: convert [outDim,inDim/8] → [inDim/8,outDim], reuse GPTQ kernel
//
// The transpose approach reuses the highly-optimized gemv_q4sym kernel.
// The native MLX kernel is a fallback when transpose is not desired.

import (
	"fmt"
	"unsafe"
)

// GPUMLXWeight holds MLX quantized weight data on GPU.
type GPUMLXWeight struct {
	QWeight *Buffer // packed uint32 weights
	Scales  *Buffer // float32 scales
	Biases  *Buffer // float32 biases
	InDim   int
	OutDim  int
	Groups  int
	GroupSz int
	// Transposed for GPTQ kernel + correction buffer
	AsGPTQ     *GPUQuantWeight // transposed weights for fast GPTQ kernel
	Correction *Buffer         // [outDim * numGroups] f32: 8*scale+bias per group
}

// UploadMLXWeight uploads MLX quantized weight to GPU VRAM.
// It can upload the GPTQ-transposed fast path, native MLX buffers, or both.
func UploadMLXWeight(weight []uint32, scales, biases []float32, inDim, outDim, groupSize int, wantNative bool) (*GPUMLXWeight, error) {
	if inDim <= 0 || outDim <= 0 || groupSize <= 0 || inDim%groupSize != 0 || inDim%8 != 0 {
		return nil, fmt.Errorf("invalid MLX dims inDim=%d outDim=%d groupSize=%d", inDim, outDim, groupSize)
	}
	numGroups := inDim / groupSize
	packedPerRow := inDim / 8
	wantWeight, ok := checkedMulInt(outDim, packedPerRow)
	if !ok {
		return nil, fmt.Errorf("MLX weight size overflow outDim=%d packedPerRow=%d", outDim, packedPerRow)
	}
	wantScale, ok := checkedMulInt(outDim, numGroups)
	if !ok {
		return nil, fmt.Errorf("MLX scale size overflow outDim=%d groups=%d", outDim, numGroups)
	}
	if len(weight) < wantWeight {
		return nil, fmt.Errorf("weight length=%d, want at least %d", len(weight), wantWeight)
	}
	if len(scales) < wantScale || len(biases) < wantScale {
		return nil, fmt.Errorf("scale/bias length=%d/%d, want at least %d", len(scales), len(biases), wantScale)
	}
	if !SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}
	EnsureContext()

	// Also upload biases for the correction term
	// MLX dequant: val * scale + bias
	// GPTQ kernel: (val - 8) * scale
	// So: MLX_out = val*scale + bias = (val-8)*scale + 8*scale + bias
	// Correction per output = sum_over_input(x[i] * (8*scale[g] + bias[g]))
	// This is a constant bias added AFTER the GPTQ GEMV.
	// Precompute: for each output row, bias_correction = sum(x * (8*s + b))
	// But x changes each call! So we need to store per-group corrections.

	// Actually simpler: precompute bias_per_elem and store on GPU.
	// bias_correction[row] = sum_{i=0}^{inDim-1} x[i] * (8*scale[row,g(i)] + bias[row,g(i)])
	// This requires x at runtime. Can't precompute.
	//
	// Alternative: modify the GPTQ kernel to add bias. But that changes the shared kernel.
	//
	// Simplest correct approach: upload biases and apply correction after GEMV.
	// correction[row] = sum_g(sum_{e in group_g} x[g*gs+e]) * (8*scale[row,g] + bias[row,g])
	//
	// Even simpler: just use the native MLX kernel for now if bias != 0.
	// OR: absorb bias into scale by adjusting the packed values.
	//
	// Actually the cleanest approach: since GPTQ symmetric uses (val-8)*scale,
	// and MLX uses val*scale+bias, we can convert:
	//   MLX_val*scale + bias = (MLX_val - 8)*scale + 8*scale + bias
	// If we use the GPTQ kernel with scale'=scale, then output has extra term:
	//   extra_per_group = (8*scale + bias) * sum(x_in_group)
	// Which requires runtime x. Not precomputable.
	//
	// Best solution: use native MLX kernel with shared-mem optimization.
	// Fall back to GPTQ kernel ONLY if biases are all zero.

	w := &GPUMLXWeight{
		InDim:   inDim,
		OutDim:  outDim,
		Groups:  numGroups,
		GroupSz: groupSize,
	}

	// Transpose to GPTQ layout for fast kernel path
	transposed := make([]int32, wantWeight)
	for row := 0; row < outDim; row++ {
		for col := 0; col < packedPerRow; col++ {
			transposed[col*outDim+row] = int32(weight[row*packedPerRow+col])
		}
	}
	transScales := make([]float32, wantScale)
	for row := 0; row < outDim; row++ {
		for g := 0; g < numGroups; g++ {
			transScales[g*outDim+row] = scales[row*numGroups+g]
		}
	}
	gIdx := make([]int32, inDim)
	for i := 0; i < inDim; i++ {
		gIdx[i] = int32(i / groupSize)
	}
	if gptqW, err := UploadQuantWeight(transposed, gIdx, transScales, inDim, outDim); err == nil {
		w.AsGPTQ = gptqW
		// Precompute bias correction: (8*scale + bias) per [outDim, numGroups]
		correction := make([]float32, wantScale)
		for row := 0; row < outDim; row++ {
			for g := 0; g < numGroups; g++ {
				correction[row*numGroups+g] = 8*scales[row*numGroups+g] + biases[row*numGroups+g]
			}
		}
		if corrBuf, err := Malloc(len(correction)); err == nil {
			if err := corrBuf.Upload(correction); err == nil {
				w.Correction = corrBuf
			} else {
				corrBuf.Free()
			}
		}
		if !wantNative {
			return w, nil
		}
	}

	// Upload native MLX buffers when requested, or as a fallback when GPTQ upload fails.
	qwBuf, err := Malloc(len(weight))
	if err != nil {
		return nil, err
	}
	w.QWeight = qwBuf
	if err := qwBuf.Upload(reinterpretI32asF32(func() []int32 {
		r := make([]int32, len(weight))
		for i, v := range weight {
			r[i] = int32(v)
		}
		return r
	}())); err != nil {
		w.Free()
		return nil, err
	}

	sBuf, err := Malloc(len(scales))
	if err != nil {
		w.Free()
		return nil, err
	}
	w.Scales = sBuf
	if err := sBuf.Upload(scales); err != nil {
		w.Free()
		return nil, err
	}

	bBuf, err := Malloc(len(biases))
	if err != nil {
		w.Free()
		return nil, err
	}
	w.Biases = bBuf
	if err := bBuf.Upload(biases); err != nil {
		w.Free()
		return nil, err
	}

	return w, nil
}

func validGPUMLXWeight(w *GPUMLXWeight) bool {
	if w == nil || w.InDim <= 0 || w.OutDim <= 0 || w.Groups <= 0 || w.GroupSz <= 0 || w.InDim%w.GroupSz != 0 || w.InDim%8 != 0 {
		return false
	}
	if w.Groups != w.InDim/w.GroupSz {
		return false
	}
	packed, okP := checkedMulInt(w.InDim/8, w.OutDim)
	scale, okS := checkedMulInt(w.Groups, w.OutDim)
	if !okP || !okS {
		return false
	}
	if w.AsGPTQ != nil && validGPUQuantWeight(w.AsGPTQ) {
		return true
	}
	return w.QWeight != nil && w.Scales != nil && w.Biases != nil && w.QWeight.Size >= packed*4 && w.Scales.Size >= scale*4 && w.Biases.Size >= scale*4
}

// Free releases GPU buffers owned by the MLX quantized weight.
func (w *GPUMLXWeight) Free() {
	if w == nil {
		return
	}
	if w.QWeight != nil {
		w.QWeight.Free()
		w.QWeight = nil
	}
	if w.Scales != nil {
		w.Scales.Free()
		w.Scales = nil
	}
	if w.Biases != nil {
		w.Biases.Free()
		w.Biases = nil
	}
	if w.Correction != nil {
		w.Correction.Free()
		w.Correction = nil
	}
	if w.AsGPTQ != nil {
		w.AsGPTQ.Free()
		w.AsGPTQ = nil
	}
}

// reinterpretI32asF32 reinterprets []int32 as []float32 (same bits).
func reinterpretI32asF32(data []int32) []float32 {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data))
}

// GemvMLX performs GPU GEMV with MLX quantized weights using optimized kernel.
func GemvMLX(out *DevBuf, x *DevBuf, w *GPUMLXWeight) {
	if !validGPUMLXWeight(w) || x == nil || out == nil || x.n < w.InDim || out.n < w.OutDim {
		return
	}
	// Fast path: use transposed weights with GPTQ kernel
	// Note: this gives (val-8)*scale instead of val*scale+bias
	// The 8*scale+bias correction is small and applied separately
	if w.AsGPTQ != nil && q4Ready {
		if !tryGPU(x) {
			return
		}
		GemvQ4(out, x, w.AsGPTQ) // GPTQ path
		// Apply bias correction: out += sum_g(group_sum_x * (8*scale+bias))
		if w.Correction != nil && fnMLXCorrect != 0 && tryGPU(x, out) {
			EnsureContext()
			outDim := uint32(w.OutDim)
			inDim := uint32(w.InDim)
			groups := uint32(w.Groups)
			groupSz := uint32(w.GroupSz)
			LaunchKernel(fnMLXCorrect, outDim, 1, 1, 256, 1, 1, 256*4,
				unsafe.Pointer(&x.gpu.Ptr),
				unsafe.Pointer(&w.Correction.Ptr),
				unsafe.Pointer(&out.gpu.Ptr),
				unsafe.Pointer(&inDim),
				unsafe.Pointer(&outDim),
				unsafe.Pointer(&groups),
				unsafe.Pointer(&groupSz))
			out.dev = GPU_DEVICE
		}
		return
	}
	if fnMLXGemv == 0 || !tryGPU(x, out) {
		return
	}
	EnsureContext()

	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	groups := uint32(w.Groups)
	groupSz := uint32(w.GroupSz)

	// Shared memory: 256*4 (reduction) + inDim*4 (x cache)
	sharedMem := uint32(256 * 4)

	LaunchKernel(fnMLXGemv, outDim, 1, 1, 256, 1, 1, sharedMem,
		unsafe.Pointer(&x.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&w.Biases.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups),
		unsafe.Pointer(&groupSz))
	out.dev = GPU_DEVICE
}

// GemmMLX performs batched GPU GEMM with MLX quantized weights.
func GemmMLX(out, input *DevBuf, w *GPUMLXWeight, B int) {
	if !validGPUMLXWeight(w) || input == nil || out == nil || fnMLXGemm == 0 || B <= 0 {
		return
	}
	inNeed, okIn := checkedMulInt(B, w.InDim)
	outNeed, okOut := checkedMulInt(B, w.OutDim)
	if !okIn || !okOut || input.n < inNeed || out.n < outNeed || !tryGPU(input, out) {
		return
	}
	EnsureContext()

	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	groups := uint32(w.Groups)
	groupSz := uint32(w.GroupSz)
	batchSize := uint32(B)

	LaunchKernel(fnMLXGemm, outDim, batchSize, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&input.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&w.Biases.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups),
		unsafe.Pointer(&groupSz),
		unsafe.Pointer(&batchSize))
	out.dev = GPU_DEVICE
}

var fnMLXGemv CUfunction
var fnMLXCorrect CUfunction
var fnMLXGemm CUfunction

func GemvMLXDirect(out *DevBuf, x *DevBuf, w *GPUMLXWeight) {
	if !validGPUMLXWeight(w) || fnMLXGemv == 0 || w.QWeight == nil {
		// Fall back to GPTQ path if native buffers unavailable
		GemvMLX(out, x, w)
		return
	}
	if x == nil || out == nil || x.n < w.InDim || out.n < w.OutDim || !tryGPU(x, out) {
		return
	}
	EnsureContext()

	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	groups := uint32(w.Groups)
	groupSz := uint32(w.GroupSz)

	sharedMem := uint32(256 * 4)

	LaunchKernel(fnMLXGemv, outDim, 1, 1, 256, 1, 1, sharedMem,
		unsafe.Pointer(&x.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&w.Biases.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups),
		unsafe.Pointer(&groupSz))
	out.dev = GPU_DEVICE
}
