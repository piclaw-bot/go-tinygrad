package model

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
func (cp FloatKVCheckpoint) Restore(kvCacheK, kvCacheV [][]float32) {
	for i, n := range cp.KLen {
		if i < len(kvCacheK) && n <= len(kvCacheK[i]) {
			kvCacheK[i] = kvCacheK[i][:n]
		}
	}
	for i, n := range cp.VLen {
		if i < len(kvCacheV) && n <= len(kvCacheV[i]) {
			kvCacheV[i] = kvCacheV[i][:n]
		}
	}
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
func (c *CompressedKVCache) Restore(cp CompressedKVCheckpoint) {
	if c == nil || !cp.valid {
		return
	}
	c.FullK = append(c.FullK[:0], cp.fullK...)
	c.FullV = append(c.FullV[:0], cp.fullV...)
	if cp.compressedKLen <= len(c.CompressedK) {
		c.CompressedK = c.CompressedK[:cp.compressedKLen]
	}
	if cp.compressedVLen <= len(c.CompressedV) {
		c.CompressedV = c.CompressedV[:cp.compressedVLen]
	}
	c.scratchK = c.scratchK[:0]
	c.scratchV = c.scratchV[:0]
	c.seqLen = cp.seqLen
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
func RestoreCompressedKV(caches []*CompressedKVCache, cp []CompressedKVCheckpoint) {
	for i, c := range caches {
		if i < len(cp) {
			c.Restore(cp[i])
		}
	}
}
