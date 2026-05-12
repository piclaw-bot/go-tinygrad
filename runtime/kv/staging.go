package kv

import "fmt"

// FloatKVCheckpoint records per-layer lengths for an uncompressed float32 KV cache.
// Restoring truncates candidate K/V appends without copying existing cache data.
type FloatKVCheckpoint struct {
	KLen []int
	VLen []int
}

// CheckpointFloatKV records the current lengths of the uncompressed per-layer KV cache.
func CheckpointFloatKV(kvCacheK, kvCacheV [][]float32) FloatKVCheckpoint {
	cp := FloatKVCheckpoint{
		KLen: make([]int, len(kvCacheK)),
		VLen: make([]int, len(kvCacheV)),
	}
	for i := range kvCacheK {
		cp.KLen[i] = len(kvCacheK[i])
	}
	for i := range kvCacheV {
		cp.VLen[i] = len(kvCacheV[i])
	}
	return cp
}

// Restore truncates the uncompressed per-layer KV cache to this checkpoint.
func (cp FloatKVCheckpoint) Restore(kvCacheK, kvCacheV [][]float32) error {
	return cp.KeepAppended(kvCacheK, kvCacheV, nil, 0)
}

// KeepAppended keeps the first keepTokens staged tokens after this checkpoint
// and discards any later staged candidate tokens. kvDims is the per-layer K/V
// vector width; zero means that layer did not append K/V during verification.
func (cp FloatKVCheckpoint) KeepAppended(kvCacheK, kvCacheV [][]float32, kvDims []int, keepTokens int) error {
	if keepTokens < 0 {
		return fmt.Errorf("keepTokens=%d must be >= 0", keepTokens)
	}
	for i, n := range cp.KLen {
		if i >= len(kvCacheK) {
			continue
		}
		kvDim := kvDimAt(kvDims, i)
		if kvDim < 0 {
			return fmt.Errorf("layer %d K kvDim=%d must be >= 0", i, kvDim)
		}
		target, ok := checkedAppendTarget(n, keepTokens, kvDim)
		if !ok || target > len(kvCacheK[i]) {
			return fmt.Errorf("layer %d K target len=%d exceeds current len=%d", i, target, len(kvCacheK[i]))
		}
		kvCacheK[i] = kvCacheK[i][:target]
	}
	for i, n := range cp.VLen {
		if i >= len(kvCacheV) {
			continue
		}
		kvDim := kvDimAt(kvDims, i)
		if kvDim < 0 {
			return fmt.Errorf("layer %d V kvDim=%d must be >= 0", i, kvDim)
		}
		target, ok := checkedAppendTarget(n, keepTokens, kvDim)
		if !ok || target > len(kvCacheV[i]) {
			return fmt.Errorf("layer %d V target len=%d exceeds current len=%d", i, target, len(kvCacheV[i]))
		}
		kvCacheV[i] = kvCacheV[i][:target]
	}
	return nil
}

func kvDimAt(kvDims []int, i int) int {
	if i >= 0 && i < len(kvDims) {
		return kvDims[i]
	}
	return 0
}

func checkedAppendTarget(base, keepTokens, kvDim int) (int, bool) {
	if base < 0 || keepTokens < 0 || kvDim < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if kvDim != 0 && keepTokens > maxInt/kvDim {
		return 0, false
	}
	add := keepTokens * kvDim
	if base > maxInt-add {
		return 0, false
	}
	return base + add, true
}

// CompressedKVCheckpoint records enough state to restore a compressed KV cache.
// Full-precision residual rows are copied because appending past the residual
// window can compress and drop the oldest full row, so length-only rollback is
// not sufficient for TurboQuant-backed caches.
type CompressedKVCheckpoint struct {
	valid          bool
	fullK          []float32
	fullV          []float32
	compressedKLen int
	compressedVLen int
	seqLen         int
}

// Checkpoint records a restorable point for this compressed KV cache.
func (c *CompressedKVCache) Checkpoint() CompressedKVCheckpoint {
	if c == nil {
		return CompressedKVCheckpoint{}
	}
	return CompressedKVCheckpoint{
		valid:          true,
		fullK:          append([]float32(nil), c.FullK...),
		fullV:          append([]float32(nil), c.FullV...),
		compressedKLen: len(c.CompressedK),
		compressedVLen: len(c.CompressedV),
		seqLen:         c.seqLen,
	}
}

// Restore rolls this compressed KV cache back to a previous checkpoint.
func (c *CompressedKVCache) Restore(cp CompressedKVCheckpoint) error {
	return c.KeepAppended(cp, 0)
}

// KeepAppended keeps the first keepTokens staged tokens after this checkpoint
// and discards later candidate tokens. This works even when candidate appends
// crossed the residual window and triggered TurboQuant compression. Kept staged
// rows are replayed through Append after rollback, so if they again cross the
// residual window they may be quantized once more by the cache policy.
func (c *CompressedKVCache) KeepAppended(cp CompressedKVCheckpoint, keepTokens int) error {
	if c == nil || !cp.valid {
		return nil
	}
	if keepTokens < 0 {
		return fmt.Errorf("keepTokens=%d must be >= 0", keepTokens)
	}
	targetSeq, ok := checkedAppendTarget(cp.seqLen, keepTokens, 1)
	if !ok {
		return fmt.Errorf("compressed KV checkpoint+keep overflows")
	}
	if c.seqLen < targetSeq {
		return fmt.Errorf("compressed KV seqLen=%d shorter than checkpoint+keep=%d", c.seqLen, targetSeq)
	}
	if cp.compressedKLen < 0 || cp.compressedVLen < 0 {
		return fmt.Errorf("checkpoint compressed lengths K/V=%d/%d must be >= 0", cp.compressedKLen, cp.compressedVLen)
	}

	var keepK, keepV []float32
	if keepTokens > 0 {
		if c.kvDim <= 0 {
			return fmt.Errorf("compressed KV kvDim=%d must be > 0 when keeping tokens", c.kvDim)
		}
		allK := c.GetK()
		allV := c.GetV()
		start, ok := checkedAppendTarget(0, cp.seqLen, c.kvDim)
		if !ok {
			return fmt.Errorf("compressed KV keep start overflows")
		}
		end, ok := checkedAppendTarget(start, keepTokens, c.kvDim)
		if !ok {
			return fmt.Errorf("compressed KV keep end overflows")
		}
		if start > len(allK) || start > len(allV) || end > len(allK) || end > len(allV) {
			return fmt.Errorf("compressed KV keep range [%d:%d] exceeds K/V lengths %d/%d", start, end, len(allK), len(allV))
		}
		keepK = append([]float32(nil), allK[start:end]...)
		keepV = append([]float32(nil), allV[start:end]...)
	}

	if cp.compressedKLen > len(c.CompressedK) || cp.compressedVLen > len(c.CompressedV) {
		return fmt.Errorf("checkpoint compressed lengths K/V=%d/%d exceed current %d/%d", cp.compressedKLen, cp.compressedVLen, len(c.CompressedK), len(c.CompressedV))
	}
	c.FullK = append(c.FullK[:0], cp.fullK...)
	c.FullV = append(c.FullV[:0], cp.fullV...)
	// Compressed entries are append-only; a checkpoint only needs the previous
	// lengths because candidate appends never mutate existing compressed entries.
	c.CompressedK = c.CompressedK[:cp.compressedKLen]
	c.CompressedV = c.CompressedV[:cp.compressedVLen]
	c.scratchK = c.scratchK[:0]
	c.scratchV = c.scratchV[:0]
	c.seqLen = cp.seqLen

	for i := 0; i < keepTokens; i++ {
		start := i * c.kvDim
		end := start + c.kvDim
		c.Append(keepK[start:end], keepV[start:end])
	}
	return nil
}

// CheckpointCompressedKV records checkpoints for all non-nil compressed layer caches.
func CheckpointCompressedKV(caches []*CompressedKVCache) []CompressedKVCheckpoint {
	cp := make([]CompressedKVCheckpoint, len(caches))
	for i, c := range caches {
		cp[i] = c.Checkpoint()
	}
	return cp
}

// RestoreCompressedKV restores all compressed layer caches from checkpoints.
func RestoreCompressedKV(caches []*CompressedKVCache, cp []CompressedKVCheckpoint) error {
	return KeepCompressedKVAppended(caches, cp, 0)
}

// KeepCompressedKVAppended keeps the first keepTokens staged positions for all
// compressed layer caches and discards later candidate positions.
func KeepCompressedKVAppended(caches []*CompressedKVCache, cp []CompressedKVCheckpoint, keepTokens int) error {
	for i, c := range caches {
		if i < len(cp) {
			if err := c.KeepAppended(cp[i], keepTokens); err != nil {
				return fmt.Errorf("layer %d: %w", i, err)
			}
		}
	}
	return nil
}
