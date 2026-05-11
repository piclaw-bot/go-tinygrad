package gpu

// Batched Q4 GEMM: process multiple tokens in one kernel launch.
//
// Standard GEMV: [1 × inDim] × [inDim × outDim] → [1 × outDim]
// Batched GEMM:  [B × inDim] × [inDim × outDim] → [B × outDim]
//
// For prefill, B = prompt length. Each row reads the SAME weight matrix,
// so weight data is fetched once from VRAM and reused across all B rows.
// This turns memory-bound GEMV into compute-bound GEMM.
//
// The kernel dequantizes INT4 weights on-the-fly (same as gemv_q4sym)
// but accumulates B dot products per weight column.

import (
	"fmt"
	"unsafe"
)

// GemmQ4 performs batched matrix multiply: out[B×outDim] = input[B×inDim] × W_q4[inDim×outDim]
// where W is INT4 quantized with group scales.
func GemmQ4(out, input *DevBuf, w *GPUQuantWeight, B int) {
	if !q4Ready || w == nil || B <= 0 {
		return
	}
	EnsureContext()

	batchSize := uint32(B)
	inDim := uint32(w.InDim)
	outDim := uint32(w.OutDim)
	groups := uint32(w.Groups)

	input.ToGPU()
	out.ToGPU()

	// Grid: one block per (output column, batch row) pair
	// Block: 256 threads, each handles part of the dot product
	gridX := outDim
	gridY := batchSize

	LaunchKernel(fnGemmQ4, gridX, gridY, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&input.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.GIdx.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups),
		unsafe.Pointer(&batchSize),
	)
	out.dev = GPU_DEVICE
}

var fnGemmQ4 CUfunction

func init() {
	// Will be extracted from mega module
}

// BatchGEMMReady returns true if batched GEMM kernel is available.
func BatchGEMMReady() bool {
	return fnGemmQ4 != 0
}

// GemvQ4OrGemm dispatches to GEMV (B=1) or batched GEMM (B>1).
func GemvQ4OrGemm(out, input *DevBuf, w *GPUQuantWeight, B int) {
	if B <= 1 {
		GemvQ4(out, input, w)
	} else {
		if fnGemmQ4 != 0 {
			GemmQ4(out, input, w, B)
		} else {
			// Fallback: sequential GEMV
			for b := 0; b < B; b++ {
				// This would need sub-buffer support
				fmt.Printf("[gpu] batch GEMM not available, falling back to sequential\n")
				return
			}
		}
	}
}
