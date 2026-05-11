package gpu

// LM head GEMV: optimized kernel for [vocab × h] × [h] → [vocab]
//
// Each block computes one output element (one row's dot product).
// 256 threads per block, shared memory tree reduction.
// This is much faster than SGEMM for N=1 (vector) cases.

import (
	"unsafe"
)

// DevLMHead computes logits[vocab] = W[vocab,h] · x[h]
// Uses a dedicated kernel optimized for large M (vocab) and small N (1).
func DevLMHead(logits, x, W *DevBuf, vocab, h int) {
	if logits == nil || x == nil || W == nil || vocab <= 0 || h <= 0 || logits.n < vocab || x.n < h || W.n < vocab*h {
		return
	}
	if !kernelsLoaded || fnLMHead == 0 || !tryGPU(x, W, logits) {
		DevGemv(logits, x, W, vocab, h)
		return
	}
	EnsureContext()

	v := uint32(vocab)
	dim := uint32(h)

	// Grid: vocab may exceed 65535 max blocks in x. Use 2D grid.
	gridX := uint32(vocab)
	gridY := uint32(1)
	if gridX > 65535 {
		gridY = (gridX + 65534) / 65535
		gridX = 65535
	}

	LaunchKernel(fnLMHead, gridX, gridY, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&W.gpu.Ptr),
		unsafe.Pointer(&x.gpu.Ptr),
		unsafe.Pointer(&logits.gpu.Ptr),
		unsafe.Pointer(&v),
		unsafe.Pointer(&dim))
	logits.dev = GPU_DEVICE
}

var fnLMHead CUfunction
