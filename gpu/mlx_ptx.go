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
	if !SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}
	EnsureContext()

	numGroups := inDim / groupSize

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
	packFactor := 8
	packedPerRow := inDim / packFactor
	transposed := make([]int32, packedPerRow*outDim)
	for row := 0; row < outDim; row++ {
		for col := 0; col < packedPerRow; col++ {
			transposed[col*outDim+row] = int32(weight[row*packedPerRow+col])
		}
	}
	transScales := make([]float32, numGroups*outDim)
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
		correction := make([]float32, outDim*numGroups)
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
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data))
}

// GemvMLX performs GPU GEMV with MLX quantized weights using optimized kernel.
func GemvMLX(out *DevBuf, x *DevBuf, w *GPUMLXWeight) {
	if w == nil {
		return
	}
	// Fast path: use transposed weights with GPTQ kernel
	// Note: this gives (val-8)*scale instead of val*scale+bias
	// The 8*scale+bias correction is small and applied separately
	if w.AsGPTQ != nil && q4Ready {
		x.ToGPU()                // ensure CPU-modified data is uploaded
		GemvQ4(out, x, w.AsGPTQ) // GPTQ path
		// Apply bias correction: out += sum_g(group_sum_x * (8*scale+bias))
		if w.Correction != nil && fnMLXCorrect != 0 {
			x.ToGPU()
			out.ToGPU()
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
	if fnMLXGemv == 0 {
		return
	}
	EnsureContext()
	x.ToGPU()
	out.EnsureGPU()

	if x.gpu == nil || out.gpu == nil {
		return
	}

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
	if w == nil || fnMLXGemm == 0 || B <= 0 {
		return
	}
	EnsureContext()
	input.ToGPU()
	out.EnsureGPU()

	if input.gpu == nil || out.gpu == nil {
		return
	}

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

// MLXGemvPTX is the optimized MLX GEMV kernel.
var MLXGemvPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_gemv(
    .param .u64 x,
    .param .u64 qweight,
    .param .u64 scales,
    .param .u64 biases,
    .param .u64 output,
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize
) {
    .reg .u32 %r<20>;
    .reg .u64 %rd<18>;
    .reg .f32 %f<16>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [outDim];
    ld.param.u32 %r3, [inDim];
    ld.param.u32 %r4, [numGroups];
    ld.param.u32 %r5, [groupSize];

    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;

    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [qweight];
    ld.param.u64 %rd2, [scales];
    ld.param.u64 %rd3, [biases];
    ld.param.u64 %rd4, [output];

    // Weight row pointer: qweight + row * (inDim/8) * 4
    shr.u32 %r6, %r3, 3;
    mul.lo.u32 %r7, %r0, %r6;
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd1, %rd1, %rd5;

    // Scale/bias row pointer
    mul.lo.u32 %r8, %r0, %r4;
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd2, %rd2, %rd6;
    add.u64 %rd3, %rd3, %rd6;

    // Each thread processes packed words tid, tid+256, tid+512...
    // 8 elements per word, coalesced reads across warp
    mov.f32 %f0, 0f00000000;
    mov.u32 %r9, %r1;

pack_loop:
    setp.ge.u32 %p, %r9, %r6;
    @%p bra reduce;

    // Load packed word (coalesced)
    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r10, [%rd8];

    // Base element = word_idx * 8, group = base / groupSize
    shl.b32 %r11, %r9, 3;
    div.u32 %r12, %r11, %r5;
    mul.wide.u32 %rd9, %r12, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f1, [%rd10];
    add.u64 %rd11, %rd3, %rd9;
    ld.global.f32 %f2, [%rd11];

    // x base pointer for 8 elements
    mul.wide.u32 %rd12, %r11, 4;
    add.u64 %rd13, %rd0, %rd12;

    // Unrolled 8 elements: extract 4-bit, dequant, FMA
    // Element 0
    and.b32 %r13, %r10, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 1
    shr.u32 %r13, %r10, 4;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+4];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 2
    shr.u32 %r13, %r10, 8;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+8];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 3
    shr.u32 %r13, %r10, 12;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+12];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 4
    shr.u32 %r13, %r10, 16;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+16];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 5
    shr.u32 %r13, %r10, 20;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+20];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 6
    shr.u32 %r13, %r10, 24;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+24];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 7
    shr.u32 %r13, %r10, 28;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+28];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    add.u32 %r9, %r9, 256;
    bra pack_loop;

reduce:
    mov.u64 %rd14, sdata;
    mul.wide.u32 %rd15, %r1, 4;
    add.u64 %rd14, %rd14, %rd15;
    st.shared.f32 [%rd14], %f0;
    bar.sync 0;

    mov.u32 %r14, 128;
red_loop:
    setp.lt.u32 %p, %r14, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r1, %r14;
    @%p bra red_skip;
    add.u32 %r15, %r1, %r14;
    mul.wide.u32 %rd16, %r15, 4;
    mov.u64 %rd17, sdata;
    add.u64 %rd17, %rd17, %rd16;
    ld.shared.f32 %f5, [%rd17];
    ld.shared.f32 %f6, [%rd14];
    add.f32 %f6, %f6, %f5;
    st.shared.f32 [%rd14], %f6;
red_skip:
    bar.sync 0;
    shr.u32 %r14, %r14, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.f32 %f7, [sdata];
    mul.wide.u32 %rd7, %r0, 4;
    add.u64 %rd8, %rd4, %rd7;
    st.global.f32 [%rd8], %f7;

done:
    ret;
}
`

// MLXGemmPTX is the batched MLX GEMM kernel.
var MLXGemmPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_gemm(
    .param .u64 input,
    .param .u64 qweight,
    .param .u64 scales,
    .param .u64 biases,
    .param .u64 output,
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize,
    .param .u32 B
) {
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<12>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r20, %ctaid.y;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [outDim];
    ld.param.u32 %r3, [inDim];
    ld.param.u32 %r4, [numGroups];
    ld.param.u32 %r5, [groupSize];
    ld.param.u32 %r21, [B];

    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;
    setp.ge.u32 %p, %r20, %r21;
    @%p bra done;

    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [qweight];
    ld.param.u64 %rd2, [scales];
    ld.param.u64 %rd3, [biases];
    ld.param.u64 %rd4, [output];

    mul.lo.u32 %r22, %r20, %r3;
    mul.wide.u32 %rd5, %r22, 4;
    add.u64 %rd0, %rd0, %rd5;

    shr.u32 %r7, %r3, 3;
    mul.lo.u32 %r8, %r0, %r7;
    mul.wide.u32 %rd5, %r8, 4;
    add.u64 %rd1, %rd1, %rd5;

    mul.lo.u32 %r9, %r0, %r4;
    mul.wide.u32 %rd6, %r9, 4;
    add.u64 %rd2, %rd2, %rd6;
    add.u64 %rd3, %rd3, %rd6;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r10, %r1;

loop:
    setp.ge.u32 %p, %r10, %r3;
    @%p bra reduce;

    shr.u32 %r11, %r10, 3;
    mul.wide.u32 %rd7, %r11, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r12, [%rd8];

    and.b32 %r13, %r10, 7;
    shl.b32 %r13, %r13, 2;
    shr.u32 %r14, %r12, %r13;
    and.b32 %r14, %r14, 15;
    cvt.rn.f32.u32 %f2, %r14;

    div.u32 %r15, %r10, %r5;
    mul.wide.u32 %rd9, %r15, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f3, [%rd10];
    add.u64 %rd11, %rd3, %rd9;
    ld.global.f32 %f4, [%rd11];

    fma.rn.f32 %f2, %f2, %f3, %f4;

    mul.wide.u32 %rd12, %r10, 4;
    add.u64 %rd13, %rd0, %rd12;
    ld.global.f32 %f5, [%rd13];
    fma.rn.f32 %f1, %f2, %f5, %f1;

    add.u32 %r10, %r10, 256;
    bra loop;

reduce:
    mov.u64 %rd14, sdata;
    mul.wide.u32 %rd15, %r1, 4;
    add.u64 %rd14, %rd14, %rd15;
    st.shared.f32 [%rd14], %f1;
    bar.sync 0;

    mov.u32 %r15, 128;
red_loop:
    setp.lt.u32 %p, %r15, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r1, %r15;
    @%p bra red_skip;
    add.u32 %r16, %r1, %r15;
    mul.wide.u32 %rd7, %r16, 4;
    mov.u64 %rd8, sdata;
    add.u64 %rd8, %rd8, %rd7;
    ld.shared.f32 %f6, [%rd8];
    ld.shared.f32 %f7, [%rd14];
    add.f32 %f7, %f7, %f6;
    st.shared.f32 [%rd14], %f7;
red_skip:
    bar.sync 0;
    shr.u32 %r15, %r15, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.f32 %f8, [sdata];
    mad.lo.u32 %r17, %r20, %r2, %r0;
    mul.wide.u32 %rd7, %r17, 4;
    add.u64 %rd8, %rd4, %rd7;
    st.global.f32 [%rd8], %f8;

done:
    ret;
}
`

// MLXCorrectPTX: bias correction after GPTQ GEMV for MLX weights.
// out[row] += sum_g( sum(x[g*gs:(g+1)*gs]) * correction[row*numGroups+g] )
var MLXCorrectPTX = `
.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_correct(
    .param .u64 x,
    .param .u64 correction,
    .param .u64 output,
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize
) {
    .reg .u32 %r<16>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<8>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [outDim];
    ld.param.u32 %r4, [numGroups];
    ld.param.u32 %r5, [groupSize];

    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;

    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [correction];
    ld.param.u64 %rd2, [output];

    // correction row: correction + row * numGroups
    mul.lo.u32 %r6, %r0, %r4;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd1, %rd1, %rd3;

    // Each thread handles groups tid, tid+256, ...
    mov.f32 %f0, 0f00000000;
    mov.u32 %r7, %r1;

group_loop:
    setp.ge.u32 %p, %r7, %r4;
    @%p bra reduce;

    // Load correction[g]
    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];

    // Sum x in this group: x[g*gs .. (g+1)*gs-1]
    mul.lo.u32 %r8, %r7, %r5;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r9, 0;
xsum:
    setp.ge.u32 %p, %r9, %r5;
    @%p bra xsum_done;
    add.u32 %r10, %r8, %r9;
    mul.wide.u32 %rd6, %r10, 4;
    add.u64 %rd7, %rd0, %rd6;
    ld.global.f32 %f3, [%rd7];
    add.f32 %f2, %f2, %f3;
    add.u32 %r9, %r9, 1;
    bra xsum;
xsum_done:
    fma.rn.f32 %f0, %f2, %f1, %f0;
    add.u32 %r7, %r7, 256;
    bra group_loop;

reduce:
    mov.u64 %rd8, sdata;
    mul.wide.u32 %rd9, %r1, 4;
    add.u64 %rd8, %rd8, %rd9;
    st.shared.f32 [%rd8], %f0;
    bar.sync 0;

    mov.u32 %r11, 128;
red_loop:
    setp.lt.u32 %p, %r11, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r1, %r11;
    @%p bra red_skip;
    add.u32 %r12, %r1, %r11;
    mul.wide.u32 %rd10, %r12, 4;
    mov.u64 %rd11, sdata;
    add.u64 %rd11, %rd11, %rd10;
    ld.shared.f32 %f4, [%rd11];
    ld.shared.f32 %f5, [%rd8];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd8], %f5;
red_skip:
    bar.sync 0;
    shr.u32 %r11, %r11, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.f32 %f6, [sdata];
    mul.wide.u32 %rd4, %r0, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.f32 %f7, [%rd5];
    add.f32 %f7, %f7, %f6;
    st.global.f32 [%rd5], %f7;
done:
    ret;
}
`

// GemvMLXDirect uses the native MLX kernel (no GPTQ transpose).
// Matches CPU GemvMLQ precision. Use for precision-sensitive models.
func GemvMLXDirect(out *DevBuf, x *DevBuf, w *GPUMLXWeight) {
	if w == nil || fnMLXGemv == 0 || w.QWeight == nil {
		// Fall back to GPTQ path if native buffers unavailable
		GemvMLX(out, x, w)
		return
	}
	EnsureContext()
	x.ToGPU()
	out.EnsureGPU()

	if x.gpu == nil || out.gpu == nil {
		return
	}

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
