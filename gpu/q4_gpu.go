package gpu

// GPU-resident INT4 quantized weights for fused dequant+GEMV.

import (
	"fmt"
	"sync"
	"unsafe"
)

var (
	q4Once  sync.Once
	q4Fn    CUfunction
	q4Ready bool
)

func initQ4() { loadMegaModule() }

// GPUQuantWeight holds INT4 weights in GPU VRAM.
type GPUQuantWeight struct {
	QWeight *Buffer // [inDim/8, outDim] packed int32
	Scales  *Buffer // [numGroups, outDim] float32
	GIdx    *Buffer // [inDim] int32
	InDim   int
	OutDim  int
	Groups  int
}

// UploadQuantWeight uploads INT4 quantized weight to GPU VRAM.
func UploadQuantWeight(qweight, gIdx []int32, scales []float32, inDim, outDim int) (*GPUQuantWeight, error) {
	if inDim <= 0 || outDim <= 0 || inDim%8 != 0 {
		return nil, fmt.Errorf("invalid Q4 dims inDim=%d outDim=%d", inDim, outDim)
	}
	packRows := inDim / 8
	wantQWeight, ok := checkedMulInt(packRows, outDim)
	if !ok {
		return nil, fmt.Errorf("qweight size overflow for inDim=%d outDim=%d", inDim, outDim)
	}
	if len(qweight) < wantQWeight {
		return nil, fmt.Errorf("qweight length=%d, want at least %d", len(qweight), wantQWeight)
	}
	if len(gIdx) < inDim {
		return nil, fmt.Errorf("gIdx length=%d, want at least %d", len(gIdx), inDim)
	}
	if len(scales) == 0 || len(scales)%outDim != 0 {
		return nil, fmt.Errorf("scales length=%d is not a positive multiple of outDim=%d", len(scales), outDim)
	}
	groups := len(scales) / outDim
	if _, ok := checkedMulInt(groups, outDim); !ok {
		return nil, fmt.Errorf("scales size overflow groups=%d outDim=%d", groups, outDim)
	}
	for i := 0; i < inDim; i++ {
		g := int(gIdx[i])
		if g < 0 || g >= groups {
			return nil, fmt.Errorf("gIdx[%d]=%d out of range [0,%d)", i, g, groups)
		}
	}
	if !Q4Ready() {
		return nil, fmt.Errorf("Q4 kernel not ready")
	}

	w := &GPUQuantWeight{InDim: inDim, OutDim: outDim, Groups: groups}

	qwBuf, err := Malloc(len(qweight))
	if err != nil {
		return nil, fmt.Errorf("alloc qweight (%d): %w", len(qweight)*4, err)
	}
	w.QWeight = qwBuf
	// Upload int32 as raw bytes
	if err := qwBuf.Upload(int32ToFloat32(qweight)); err != nil {
		w.Free()
		return nil, err
	}

	scBuf, err := Malloc(len(scales))
	if err != nil {
		w.Free()
		return nil, err
	}
	w.Scales = scBuf
	if err := scBuf.Upload(scales); err != nil {
		w.Free()
		return nil, err
	}

	giBuf, err := Malloc(len(gIdx))
	if err != nil {
		w.Free()
		return nil, err
	}
	w.GIdx = giBuf
	if err := giBuf.Upload(int32ToFloat32(gIdx)); err != nil {
		w.Free()
		return nil, err
	}

	return w, nil
}

// Free releases GPU buffers owned by the quantized weight.
func (w *GPUQuantWeight) Free() {
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
	if w.GIdx != nil {
		w.GIdx.Free()
		w.GIdx = nil
	}
}

func validGPUQuantWeight(w *GPUQuantWeight) bool {
	if w == nil || w.InDim <= 0 || w.OutDim <= 0 || w.Groups <= 0 || w.InDim%8 != 0 || w.QWeight == nil || w.Scales == nil || w.GIdx == nil {
		return false
	}
	qw, okQ := checkedMulInt(w.InDim/8, w.OutDim)
	sc, okS := checkedMulInt(w.Groups, w.OutDim)
	qwBytes, errQ := checkedByteSize(qw, -1)
	scBytes, errS := checkedByteSize(sc, -1)
	giBytes, errG := checkedByteSize(w.InDim, -1)
	return okQ && okS && errQ == nil && errS == nil && errG == nil && w.QWeight.Size >= int(qwBytes) && w.Scales.Size >= int(scBytes) && w.GIdx.Size >= int(giBytes)
}

// GemvQ4 computes out[outDim] = x[inDim] @ dequant(W) on GPU.
func GemvQ4(out *DevBuf, x *DevBuf, w *GPUQuantWeight) {
	if !validGPUQuantWeight(w) || x == nil || out == nil || x.n < w.InDim || out.n < w.OutDim {
		return
	}
	if !q4Ready || !tryGPU(x, out) {
		gemvQ4CPU(out, x, w)
		return
	}

	inDim := uint32(w.InDim)
	outDim := uint32(w.OutDim)
	groups := uint32(w.Groups)

	LaunchKernel(q4Fn, (outDim+255)/256, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&w.GIdx.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups))

	out.dev = GPU_DEVICE
}

// CPU fallback for INT4 GEMV
func gemvQ4CPU(out, x *DevBuf, w *GPUQuantWeight) {
	if !validGPUQuantWeight(w) || x == nil || out == nil || x.n < w.InDim || out.n < w.OutDim {
		return
	}
	x.ToCPU()
	out.ToCPU()
	xd := x.cpu
	od := out.cpu
	// Download weight data from GPU
	qw := make([]int32, len(float32ToInt32Placeholder(w.QWeight.Size/4)))
	sc := make([]float32, w.Groups*w.OutDim)
	gi := make([]int32, w.InDim)
	if err := w.QWeight.Download(int32ToFloat32(qw)); err != nil {
		return
	}
	if err := w.Scales.Download(sc); err != nil {
		return
	}
	if err := w.GIdx.Download(int32ToFloat32(gi)); err != nil {
		return
	}

	for j := 0; j < w.OutDim; j++ {
		sum := float32(0)
		for packIdx := 0; packIdx < w.InDim/8; packIdx++ {
			packed := qw[packIdx*w.OutDim+j]
			for bit := 0; bit < 8; bit++ {
				i := packIdx*8 + bit
				qv := (packed >> (uint(bit) * 4)) & 0xF
				g := int(gi[i])
				scale := sc[g*w.OutDim+j]
				sum += xd[i] * scale * float32(qv-8)
			}
		}
		od[j] = sum
	}
}

// Helpers for int32 <-> float32 reinterpret (same bit pattern, different type)
func int32ToFloat32(data []int32) []float32 {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data))
}

func float32ToInt32Placeholder(n int) []int32 {
	if n <= 0 {
		return nil
	}
	return make([]int32, n)
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}
