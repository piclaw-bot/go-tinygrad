package model

import (
	"math"
	"testing"
)

func TestCompressedKVCacheBasic(t *testing.T) {
	headDim := 128
	numKVHeads := 4
	kvDim := numKVHeads * headDim
	tq := NewTurboQuantState(headDim, 28, DefaultTurboQuantConfig())

	cache := NewCompressedKVCache(kvDim, numKVHeads, headDim, tq, false)

	// Append 200 tokens (residual window = 128, so 72 should be compressed)
	vecs := make([][]float32, 200)
	for pos := 0; pos < 200; pos++ {
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		for i := range k {
			k[i] = float32(math.Sin(float64(pos*kvDim+i)*0.01)) * 2.0
			v[i] = float32(math.Cos(float64(pos*kvDim+i)*0.01)) * 1.5
		}
		vecs[pos] = k
		cache.Append(k, v)
	}

	t.Logf("SeqLen=%d Compressed=%d Full=%d", cache.SeqLen(), cache.CompressedCount(), cache.FullCount())
	t.Logf("Memory: %d bytes (vs %d uncompressed)", cache.MemoryBytes(), int64(200*kvDim*2*4))

	if cache.SeqLen() != 200 {
		t.Fatalf("expected 200 tokens, got %d", cache.SeqLen())
	}
	if cache.CompressedCount() != 72 {
		t.Fatalf("expected 72 compressed, got %d", cache.CompressedCount())
	}
	if cache.FullCount() != 128 {
		t.Fatalf("expected 128 full, got %d", cache.FullCount())
	}

	// Check that GetK returns correct length
	allK := cache.GetK()
	if len(allK) != 200*kvDim {
		t.Fatalf("expected %d K values, got %d", 200*kvDim, len(allK))
	}

	// Check reconstruction quality of compressed entries
	var maxErr float64
	for pos := 0; pos < 72; pos++ { // first 72 are compressed
		for i := 0; i < kvDim; i++ {
			d := math.Abs(float64(vecs[pos][i] - allK[pos*kvDim+i]))
			if d > maxErr {
				maxErr = d
			}
		}
	}
	t.Logf("Compressed K reconstruction maxErr=%.6f", maxErr)

	// Full-precision entries should be exact
	for pos := 72; pos < 200; pos++ {
		for i := 0; i < kvDim; i++ {
			d := math.Abs(float64(vecs[pos][i] - allK[pos*kvDim+i]))
			if d > 1e-6 {
				t.Fatalf("full-precision entry pos=%d i=%d differs: got %.6f want %.6f",
					pos, i, allK[pos*kvDim+i], vecs[pos][i])
			}
		}
	}
	t.Log("Full-precision entries are exact ✓")

	// Compression ratio
	origBytes := int64(200 * kvDim * 2 * 4) // K+V, float32
	ratio := float64(origBytes) / float64(cache.MemoryBytes())
	t.Logf("Compression ratio: %.1f×", ratio)
}

func TestCompressedKVCacheProtected(t *testing.T) {
	headDim := 128
	numKVHeads := 4
	kvDim := numKVHeads * headDim
	tq := NewTurboQuantState(headDim, 28, DefaultTurboQuantConfig())

	// Protected layer — should never compress
	cache := NewCompressedKVCache(kvDim, numKVHeads, headDim, tq, true)

	for pos := 0; pos < 200; pos++ {
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		cache.Append(k, v)
	}

	if cache.CompressedCount() != 0 {
		t.Fatalf("protected layer should not compress, got %d compressed", cache.CompressedCount())
	}
	t.Logf("Protected layer: %d full, 0 compressed ✓", cache.FullCount())
}
