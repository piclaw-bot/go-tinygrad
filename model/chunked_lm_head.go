package model

// Chunked GPU LM head: processes vocab projection in GPU-sized chunks.
//
// When full LM head (2.2GB) doesn't fit in VRAM, split into chunks:
//   1. Allocate a GPU buffer for chunkSize rows
//   2. Upload chunk of LM head weights
//   3. GPU GEMV for chunkSize rows
//   4. Download logits for those rows
//   5. Repeat for all chunks
//
// This trades upload bandwidth for GPU compute speed.

import (
	"github.com/rcarmo/go-pherence/gpu"
)

// chunkedGPULMHead computes logits using GPU in chunks.
// Returns true if GPU path was used.
func (g *GPUModel) chunkedGPULMHead(logits, hidden []float32, vocabSize, h int) bool {
	if g == nil || vocabSize <= 0 || h <= 0 || len(logits) < vocabSize || len(hidden) < h {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	if vocabSize > maxInt/h || len(g.lmHead) < vocabSize*h {
		return false
	}
	free, _ := gpu.MemInfo()
	if free < 64*1024*1024 { // need at least 64MB free
		return false
	}

	// Calculate chunk size: leave 32MB headroom
	if free > uint64(maxInt) {
		free = uint64(maxInt)
	}
	usable := int(free) - 32*1024*1024
	if h > maxInt/4 || usable <= 0 {
		return false
	}
	chunkRows := usable / (h * 4) // rows that fit in VRAM
	if chunkRows < 1024 {
		return false
	}
	if chunkRows > vocabSize {
		chunkRows = vocabSize
	}

	chunkElems, ok := checkedProduct(chunkRows, h)
	if !ok {
		return false
	}

	// Allocate GPU buffers for chunk
	wBuf := gpu.NewDevBuf(chunkElems)
	defer wBuf.Free()
	if err := wBuf.ToGPU(); err != nil {
		return false
	}
	outBuf := gpu.NewDevBuf(chunkRows)
	defer outBuf.Free()
	if err := outBuf.ToGPU(); err != nil {
		return false
	}
	inBuf := gpu.NewDevBuf(h)
	defer inBuf.Free()
	copy(inBuf.Data(), hidden[:h])
	inBuf.MarkDirty()
	if err := inBuf.ToGPU(); err != nil {
		return false
	}

	// Process in chunks
	for start := 0; start < vocabSize; start += chunkRows {
		end := start + chunkRows
		if end > vocabSize {
			end = vocabSize
		}
		rows := end - start

		// Upload this chunk of weights
		wData := wBuf.Data()
		copy(wData[:rows*h], g.lmHead[start*h:end*h])
		wBuf.MarkDirty()
		if err := wBuf.ToGPU(); err != nil {
			return false
		}

		// GPU GEMV: outBuf[rows] = wBuf[rows,h] · inBuf[h]
		if rows == chunkRows {
			gpu.DevLMHead(outBuf, inBuf, wBuf, rows, h)
		} else {
			// Last chunk may be smaller
			outSlice := outBuf.Slice(0, rows)
			wSlice := wBuf.Slice(0, rows*h)
			gpu.DevLMHead(outSlice, inBuf, wSlice, rows, h)
		}
		gpu.Sync()

		// Download logits
		outData := outBuf.Data()
		copy(logits[start:end], outData[:rows])
	}

	return true
}
