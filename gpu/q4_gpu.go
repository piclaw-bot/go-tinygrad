package gpu

// GPU-resident INT4 quantized weights for fused dequant+GEMV.

import (
	"fmt"
	"sync"
	"unsafe"
)

var (
	q4Once   sync.Once
	q4Fn     CUfunction
	q4Ready  bool
)

func initQ4() {
	q4Once.Do(func() {
		if !SgemmReady() {
			return
		}
		var warmPtr CUdeviceptr
		if r := cuMemAlloc(&warmPtr, 512*1024*1024); r == CUDA_SUCCESS {
			cuMemFree(warmPtr)
		}
		var err error
		q4Fn, err = LoadPTX(GemvQ4OptPTX, "gemv_q4sym")
		if err != nil {
			return
		}
		q4Ready = true
		fmt.Println("[gpu] INT4 fused dequant+GEMV kernel loaded")
	})
}

// Q4Ready returns true if the INT4 GPU kernel is available.
func Q4Ready() bool {
	initQ4()
	return q4Ready
}

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
	if !Q4Ready() {
		return nil, fmt.Errorf("Q4 kernel not ready")
	}

	groups := len(scales) / outDim

	qwBuf, err := Malloc(len(qweight))
	if err != nil {
		return nil, fmt.Errorf("alloc qweight (%d): %w", len(qweight)*4, err)
	}
	// Upload int32 as raw bytes
	qwBuf.Upload(int32ToFloat32(qweight))

	scBuf, err := Malloc(len(scales))
	if err != nil {
		return nil, err
	}
	scBuf.Upload(scales)

	giBuf, err := Malloc(len(gIdx))
	if err != nil {
		return nil, err
	}
	giBuf.Upload(int32ToFloat32(gIdx))

	return &GPUQuantWeight{
		QWeight: qwBuf,
		Scales:  scBuf,
		GIdx:    giBuf,
		InDim:   inDim,
		OutDim:  outDim,
		Groups:  groups,
	}, nil
}

// GemvQ4 computes out[outDim] = x[inDim] @ dequant(W) on GPU.
func GemvQ4(out *DevBuf, x *DevBuf, w *GPUQuantWeight) {
	if !q4Ready {
		gemvQ4CPU(out, x, w)
		return
	}

	// x already on GPU (caller uploads)
	out.EnsureGPU()

	if x.gpu == nil || out.gpu == nil {
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
	x.ToCPU()
	out.ToCPU()
	xd := x.cpu
	od := out.cpu
	// Download weight data from GPU
	qw := make([]int32, len(float32ToInt32Placeholder(w.QWeight.Size/4)))
	sc := make([]float32, w.Groups*w.OutDim)
	gi := make([]int32, w.InDim)
	w.QWeight.Download(int32ToFloat32(qw))
	w.Scales.Download(sc)
	w.GIdx.Download(int32ToFloat32(gi))

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
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data))
}

func float32ToInt32Placeholder(n int) []int32 {
	return make([]int32, n)
}
