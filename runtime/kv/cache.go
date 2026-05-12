package kv

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
	if kvDim < 0 {
		kvDim = 0
	}
	if numKVHeads < 0 {
		numKVHeads = 0
	}
	if headDim < 0 {
		headDim = 0
	}
	if numKVHeads == 0 || headDim == 0 || numKVHeads*headDim != kvDim {
		numKVHeads = 0
		headDim = 0
	}
	rw := 128
	if tq != nil {
		rw = tq.Config.ResidualWindow
	}
	if rw < 0 {
		rw = 0
	}
	capHint, ok := checkedMulInt(2048, kvDim)
	if !ok {
		capHint = 0
	}
	return &CompressedKVCache{
		FullK:          make([]float32, 0, capHint),
		FullV:          make([]float32, 0, capHint),
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
	if c == nil || c.kvDim <= 0 || len(k) != c.kvDim || len(v) != c.kvDim {
		return
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
	if c == nil || c.kvDim <= 0 || c.numKVHeads <= 0 || c.headDim <= 0 || c.numKVHeads*c.headDim != c.kvDim || c.tq == nil {
		return
	}
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
	if c == nil || c.kvDim <= 0 {
		return nil
	}
	if len(c.CompressedK) == 0 {
		if c.seqLen > 0 {
			need, ok := checkedMulInt(c.seqLen, c.kvDim)
			if ok && len(c.FullK) > need {
				return c.FullK[:need]
			}
		}
		return c.FullK
	}
	if c.tq == nil || c.numKVHeads <= 0 || c.headDim <= 0 || c.numKVHeads*c.headDim != c.kvDim {
		return c.FullK
	}
	// Decompress + concatenate into reusable scratch storage.
	need, ok := checkedMulInt(c.seqLen, c.kvDim)
	if !ok {
		return c.FullK
	}
	if cap(c.scratchK) < need {
		c.scratchK = make([]float32, 0, need)
	}
	out := c.scratchK[:0]
	bytesPerHead := (c.headDim*c.tq.Config.KeyBits + 7) / 8
	for _, entry := range c.CompressedK {
		if !compressedEntryValid(entry, c.numKVHeads, bytesPerHead) {
			return c.FullK
		}
		for h := 0; h < c.numKVHeads; h++ {
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
	if c == nil || c.kvDim <= 0 {
		return nil
	}
	if len(c.CompressedV) == 0 {
		if c.seqLen > 0 {
			need, ok := checkedMulInt(c.seqLen, c.kvDim)
			if ok && len(c.FullV) > need {
				return c.FullV[:need]
			}
		}
		return c.FullV
	}
	if c.tq == nil || c.numKVHeads <= 0 || c.headDim <= 0 || c.numKVHeads*c.headDim != c.kvDim {
		return c.FullV
	}
	need, ok := checkedMulInt(c.seqLen, c.kvDim)
	if !ok {
		return c.FullV
	}
	if cap(c.scratchV) < need {
		c.scratchV = make([]float32, 0, need)
	}
	out := c.scratchV[:0]
	bytesPerHead := (c.headDim*c.tq.Config.ValueBits + 7) / 8
	for _, entry := range c.CompressedV {
		if !compressedEntryValid(entry, c.numKVHeads, bytesPerHead) {
			return c.FullV
		}
		for h := 0; h < c.numKVHeads; h++ {
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
	if c == nil {
		return 0
	}
	return c.seqLen
}

// CompressedCount returns how many positions are compressed.
func (c *CompressedKVCache) CompressedCount() int {
	if c == nil {
		return 0
	}
	return len(c.CompressedK)
}

// FullCount returns how many positions are at full precision.
func (c *CompressedKVCache) FullCount() int {
	if c == nil || c.kvDim <= 0 {
		return 0
	}
	return len(c.FullK) / c.kvDim
}

// Reset clears the cache for reuse with a new sequence.
func (c *CompressedKVCache) Reset() {
	if c == nil {
		return
	}
	c.FullK = c.FullK[:0]
	c.FullV = c.FullV[:0]
	c.CompressedK = c.CompressedK[:0]
	c.CompressedV = c.CompressedV[:0]
	c.scratchK = c.scratchK[:0]
	c.scratchV = c.scratchV[:0]
	c.seqLen = 0
}

// MemoryBytes returns approximate memory usage (compressed + full, excluding slice headers).
func compressedEntryValid(entry compressedEntry, heads, bytesPerHead int) bool {
	if heads <= 0 || bytesPerHead <= 0 {
		return false
	}
	packedLen, ok := checkedMulInt(heads, bytesPerHead)
	return ok && len(entry.Packed) >= packedLen && len(entry.HeadVMin) >= heads && len(entry.HeadScale) >= heads
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}

func checkedAddInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if a > maxInt-b {
		return 0, false
	}
	return a + b, true
}

func maxInt64() int64 { return int64(^uint64(0) >> 1) }

func saturatingAddInt64(a, b int64) int64 {
	if a < 0 || b < 0 {
		return 0
	}
	max := maxInt64()
	if a > max-b {
		return max
	}
	return a + b
}

func saturatingMulInt64(a, b int64) int64 {
	if a < 0 || b < 0 {
		return 0
	}
	max := maxInt64()
	if b != 0 && a > max/b {
		return max
	}
	return a * b
}

func (c *CompressedKVCache) MemoryBytes() int64 {
	if c == nil {
		return 0
	}
	fullElems, ok := checkedAddInt(len(c.FullK), len(c.FullV))
	if !ok {
		return maxInt64()
	}
	full := saturatingMulInt64(int64(fullElems), 4)
	compressed := int64(0)
	entryBytes := func(e compressedEntry) int64 {
		headElems, ok := checkedAddInt(len(e.HeadVMin), len(e.HeadScale))
		if !ok {
			return maxInt64()
		}
		return saturatingAddInt64(int64(len(e.Packed)), saturatingMulInt64(int64(headElems), 4))
	}
	for _, e := range c.CompressedK {
		compressed = saturatingAddInt64(compressed, entryBytes(e))
	}
	for _, e := range c.CompressedV {
		compressed = saturatingAddInt64(compressed, entryBytes(e))
	}
	return saturatingAddInt64(full, compressed)
}
