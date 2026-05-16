package gpu

// LM head GEMV: optimized kernel for [vocab × h] × [h] → [vocab]
//
// Each block computes one output element (one row's dot product).
// 64 threads per block, shared memory tree reduction.
// This is much faster than SGEMM for N=1 (vector) cases.

import (
	"unsafe"
)

// DevLMHead computes logits[vocab] = W[vocab,h] · x[h]
// Uses a dedicated kernel optimized for large M (vocab) and small N (1).
func DevLMHead(logits, x, W *DevBuf, vocab, h int) {
	weightLen, ok := checkedMulInt(vocab, h)
	if logits == nil || x == nil || W == nil || vocab <= 0 || h <= 0 || !ok || logits.n < vocab || x.n < h || W.n < weightLen {
		return
	}
	if !kernelsLoaded || fnLMHead == 0 || !fitsUint32(vocab) || !fitsUint32(h) || !tryGPU(x, W, logits) {
		DevGemv(logits, x, W, vocab, h)
		return
	}
	EnsureContext()

	v := uint32(vocab)
	dim := uint32(h)

	// Grid: vocab may exceed 65535 max blocks in x. Use 2D grid.
	gridX := uint32(vocab)
	gridY := uint32(1)
	if vocab > 65535 {
		gy := (uint64(vocab) + 65534) / 65535
		if gy > uint64(^uint32(0)) {
			DevGemv(logits, x, W, vocab, h)
			return
		}
		gridY = uint32(gy)
		gridX = 65535
	}

	if err := LaunchKernel(fnLMHead, gridX, gridY, 1, 64, 1, 1, 64*4,
		unsafe.Pointer(&W.gpu.Ptr),
		unsafe.Pointer(&x.gpu.Ptr),
		unsafe.Pointer(&logits.gpu.Ptr),
		unsafe.Pointer(&v),
		unsafe.Pointer(&dim)); err == nil {
		logits.dev = GPU_DEVICE
		return
	}
	DevGemv(logits, x, W, vocab, h)
}

var fnLMHead CUfunction
