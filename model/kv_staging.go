package model

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
		target := n + keepTokens*kvDim
		if target > len(kvCacheK[i]) {
			return fmt.Errorf("layer %d K target len=%d exceeds current len=%d", i, target, len(kvCacheK[i]))
		}
		kvCacheK[i] = kvCacheK[i][:target]
	}
	for i, n := range cp.VLen {
		if i >= len(kvCacheV) {
			continue
		}
		kvDim := kvDimAt(kvDims, i)
		target := n + keepTokens*kvDim
		if target > len(kvCacheV[i]) {
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

// LayerKVDim returns the per-token K/V vector width appended by one layer.
// Shared-KV layers return 0 because they reuse a source layer and do not append.
func (m *LlamaModel) LayerKVDim(layerIdx int) (int, error) {
	if m == nil {
		return 0, fmt.Errorf("nil model")
	}
	if layerIdx < 0 || layerIdx >= len(m.Layers) {
		return 0, fmt.Errorf("layer index %d out of range [0,%d)", layerIdx, len(m.Layers))
	}
	layer := m.Layers[layerIdx]
	if !layer.HasKV {
		return 0, nil
	}
	if m.Config.NumKVHeads <= 0 {
		return 0, fmt.Errorf("num_key_value_heads=%d", m.Config.NumKVHeads)
	}
	headDim := m.Config.HeadDim
	if layer.HeadDimLocal > 0 {
		headDim = layer.HeadDimLocal
	}
	if headDim <= 0 {
		return 0, fmt.Errorf("layer %d head_dim=%d", layerIdx, headDim)
	}
	return m.Config.NumKVHeads * headDim, nil
}

// LayerKVDims returns per-layer K/V widths suitable for FloatKVCheckpoint
// keep-prefix commits. Layers that do not append K/V have dimension 0.
func (m *LlamaModel) LayerKVDims() ([]int, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	dims := make([]int, len(m.Layers))
	for i := range m.Layers {
		dim, err := m.LayerKVDim(i)
		if err != nil {
			return nil, err
		}
		dims[i] = dim
	}
	return dims, nil
}

// CommitAcceptedFloatKV keeps the accepted verifier KV prefix plus bonus token
// using this model's per-layer K/V widths.
func (m *LlamaModel) CommitAcceptedFloatKV(kvCacheK, kvCacheV [][]float32, cp FloatKVCheckpoint, acceptance MTPAcceptance) error {
	dims, err := m.LayerKVDims()
	if err != nil {
		return err
	}
	return CommitAcceptedFloatKV(kvCacheK, kvCacheV, cp, dims, acceptance)
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
	if c.seqLen < cp.seqLen+keepTokens {
		return fmt.Errorf("compressed KV seqLen=%d shorter than checkpoint+keep=%d", c.seqLen, cp.seqLen+keepTokens)
	}

	var keepK, keepV []float32
	if keepTokens > 0 {
		allK := c.GetK()
		allV := c.GetV()
		start := cp.seqLen * c.kvDim
		end := (cp.seqLen + keepTokens) * c.kvDim
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
