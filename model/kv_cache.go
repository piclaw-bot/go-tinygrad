package model

// CompressedKVCache wraps a per-layer KV cache with TurboQuant compression.
// Recent tokens (within the residual window) stay at full precision.
// Older tokens are compressed on demand.
type CompressedKVCache struct {
	// Full-precision storage for recent tokens
	FullK []float32 // [seqLen * kvDim] — full precision, appended per token
	FullV []float32

	// Compressed storage for older tokens
	CompressedK []compressedEntry
	CompressedV []compressedEntry

	// Reusable decompression scratch buffers. GetK/GetV return these when
	// compressed entries exist, so callers must treat returned slices as ephemeral.
	scratchK []float32
	scratchV []float32

	// Config
	kvDim          int
	numKVHeads     int
	headDim        int
	tq             *TurboQuantState
	isProtected    bool // if true, never compress this layer
	residualWindow int
	seqLen         int // total tokens stored (compressed + full)
}

type compressedEntry struct {
	Packed    []byte    // all heads' packed data concatenated
	HeadVMin  []float32 // per-head min values
	HeadScale []float32 // per-head scale values
}

// NewCompressedKVCache creates a cache for one layer.
func NewCompressedKVCache(kvDim, numKVHeads, headDim int, tq *TurboQuantState, isProtected bool) *CompressedKVCache {
	rw := 128
	if tq != nil {
		rw = tq.Config.ResidualWindow
	}
	return &CompressedKVCache{
		FullK:          make([]float32, 0, 2048*kvDim),
		FullV:          make([]float32, 0, 2048*kvDim),
		kvDim:          kvDim,
		numKVHeads:     numKVHeads,
		headDim:        headDim,
		tq:             tq,
		isProtected:    isProtected,
		residualWindow: rw,
	}
}

// Append adds a new K/V pair for the current position.
func (c *CompressedKVCache) Append(k, v []float32) {
	if len(k) != c.kvDim || len(v) != c.kvDim {
		panic("CompressedKVCache.Append: K/V vector length mismatch")
	}
	c.FullK = append(c.FullK, k...)
	c.FullV = append(c.FullV, v...)
	c.seqLen++

	// Compress old entries if we exceed the residual window
	if c.tq != nil && !c.isProtected && c.seqLen > c.residualWindow {
		c.compressOldest()
	}
}

// compressOldest moves the oldest full-precision entry to compressed storage.
func (c *CompressedKVCache) compressOldest() {
	// How many full-precision entries we have
	fullCount := len(c.FullK) / c.kvDim
	if fullCount <= c.residualWindow {
		return
	}

	// Compress per-head for the oldest entry
	// Each head's K and V vectors are compressed independently
	kVec := c.FullK[:c.kvDim]
	vVec := c.FullV[:c.kvDim]

	var ek, ev compressedEntry
	ek.Packed = make([]byte, 0)
	ev.Packed = make([]byte, 0)
	ek.HeadVMin = make([]float32, c.numKVHeads)
	ek.HeadScale = make([]float32, c.numKVHeads)
	ev.HeadVMin = make([]float32, c.numKVHeads)
	ev.HeadScale = make([]float32, c.numKVHeads)

	for h := 0; h < c.numKVHeads; h++ {
		headK := kVec[h*c.headDim : (h+1)*c.headDim]
		headV := vVec[h*c.headDim : (h+1)*c.headDim]

		pk, vMinK, scaleK := c.tq.QuantizeVector(headK, c.tq.RotationK, c.tq.CodebookK, c.tq.Config.KeyBits)
		pv, vMinV, scaleV := c.tq.QuantizeVector(headV, c.tq.RotationV, c.tq.CodebookV, c.tq.Config.ValueBits)

		ek.Packed = append(ek.Packed, pk...)
		ev.Packed = append(ev.Packed, pv...)
		ek.HeadVMin[h] = vMinK
		ek.HeadScale[h] = scaleK
		ev.HeadVMin[h] = vMinV
		ev.HeadScale[h] = scaleV
	}

	c.CompressedK = append(c.CompressedK, ek)
	c.CompressedV = append(c.CompressedV, ev)

	// Remove oldest from full-precision
	c.FullK = c.FullK[c.kvDim:]
	c.FullV = c.FullV[c.kvDim:]
}

// GetK returns the full K cache as flat []float32 for attention.
// Decompresses compressed entries on-the-fly.
func (c *CompressedKVCache) GetK() []float32 {
	if len(c.CompressedK) == 0 {
		return c.FullK
	}
	// Decompress + concatenate into reusable scratch storage.
	need := c.seqLen * c.kvDim
	if cap(c.scratchK) < need {
		c.scratchK = make([]float32, 0, need)
	}
	out := c.scratchK[:0]
	for _, entry := range c.CompressedK {
		for h := 0; h < c.numKVHeads; h++ {
			bytesPerHead := (c.headDim*c.tq.Config.KeyBits + 7) / 8
			packed := entry.Packed[h*bytesPerHead : (h+1)*bytesPerHead]
			restored := c.tq.DequantizeVector(packed, entry.HeadVMin[h], entry.HeadScale[h], c.tq.RotationK, c.tq.Config.KeyBits, c.headDim)
			out = append(out, restored...)
		}
	}
	out = append(out, c.FullK...)
	c.scratchK = out
	return out
}

// GetV returns the full V cache as flat []float32 for attention.
func (c *CompressedKVCache) GetV() []float32 {
	if len(c.CompressedV) == 0 {
		return c.FullV
	}
	need := c.seqLen * c.kvDim
	if cap(c.scratchV) < need {
		c.scratchV = make([]float32, 0, need)
	}
	out := c.scratchV[:0]
	for _, entry := range c.CompressedV {
		for h := 0; h < c.numKVHeads; h++ {
			bytesPerHead := (c.headDim*c.tq.Config.ValueBits + 7) / 8
			packed := entry.Packed[h*bytesPerHead : (h+1)*bytesPerHead]
			restored := c.tq.DequantizeVector(packed, entry.HeadVMin[h], entry.HeadScale[h], c.tq.RotationV, c.tq.Config.ValueBits, c.headDim)
			out = append(out, restored...)
		}
	}
	out = append(out, c.FullV...)
	c.scratchV = out
	return out
}

// SeqLen returns the total number of cached positions.
func (c *CompressedKVCache) SeqLen() int {
	return c.seqLen
}

// CompressedCount returns how many positions are compressed.
func (c *CompressedKVCache) CompressedCount() int {
	return len(c.CompressedK)
}

// FullCount returns how many positions are at full precision.
func (c *CompressedKVCache) FullCount() int {
	return len(c.FullK) / c.kvDim
}

// Reset clears the cache for reuse with a new sequence.
func (c *CompressedKVCache) Reset() {
	c.FullK = c.FullK[:0]
	c.FullV = c.FullV[:0]
	c.CompressedK = c.CompressedK[:0]
	c.CompressedV = c.CompressedV[:0]
	c.scratchK = c.scratchK[:0]
	c.scratchV = c.scratchV[:0]
	c.seqLen = 0
}

// MemoryBytes returns approximate memory usage (compressed + full, excluding slice headers).
func (c *CompressedKVCache) MemoryBytes() int64 {
	full := int64(len(c.FullK)+len(c.FullV)) * 4
	var compressed int64
	for _, e := range c.CompressedK {
		compressed += int64(len(e.Packed)) + int64(len(e.HeadVMin)+len(e.HeadScale))*4
	}
	for _, e := range c.CompressedV {
		compressed += int64(len(e.Packed)) + int64(len(e.HeadVMin)+len(e.HeadScale))*4
	}
	return full + compressed
}
