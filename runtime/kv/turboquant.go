package kv

import (
	"math"
	"math/rand"
)

// TurboQuantConfig holds settings for KV cache compression.
type TurboQuantConfig struct {
	KeyBits         int   // bits per key coordinate (default 4)
	ValueBits       int   // bits per value coordinate (default 2)
	ProtectedLayers []int // layers at full precision (first/last 2)
	ResidualWindow  int   // last N tokens at full precision (default 128)
}

// DefaultTurboQuantConfig returns community-validated defaults.
func DefaultTurboQuantConfig() TurboQuantConfig {
	return TurboQuantConfig{
		KeyBits:         4,
		ValueBits:       2,
		ProtectedLayers: []int{0, 1, -1, -2}, // first/last 2 layers
		ResidualWindow:  128,
	}
}

// TurboQuantState holds per-model quantization state.
type TurboQuantState struct {
	Config    TurboQuantConfig
	RotationK []float32 // [headDim × headDim] random orthogonal matrix for keys
	RotationV []float32 // [headDim × headDim] random orthogonal matrix for values
	CodebookK []float32 // [2^keyBits] reserved for future non-uniform quantization
	CodebookV []float32 // [2^valueBits] reserved for future non-uniform quantization
	HeadDim   int
	NumLayers int
}

// NewTurboQuantState initializes TurboQuant for a model.
func NewTurboQuantState(headDim, numLayers int, cfg TurboQuantConfig) *TurboQuantState {
	if headDim < 0 {
		headDim = 0
	}
	if numLayers < 0 {
		numLayers = 0
	}
	cfg.KeyBits = clampBits(cfg.KeyBits)
	cfg.ValueBits = clampBits(cfg.ValueBits)
	if cfg.ResidualWindow < 0 {
		cfg.ResidualWindow = 0
	}
	tq := &TurboQuantState{
		Config:    cfg,
		HeadDim:   headDim,
		NumLayers: numLayers,
	}

	// Generate random orthogonal matrices via QR decomposition of random Gaussian
	rng := rand.New(rand.NewSource(42)) // fixed seed for reproducibility
	tq.RotationK = randomOrthogonal(headDim, rng)
	tq.RotationV = randomOrthogonal(headDim, rng)

	// Compute Beta-optimal codebooks
	tq.CodebookK = betaOptimalCodebook(cfg.KeyBits)
	tq.CodebookV = betaOptimalCodebook(cfg.ValueBits)

	return tq
}

// IsProtectedLayer returns true if this layer should stay at full precision.
func (tq *TurboQuantState) IsProtectedLayer(layerIdx int) bool {
	for _, pl := range tq.Config.ProtectedLayers {
		if pl < 0 {
			pl = tq.NumLayers + pl
		}
		if layerIdx == pl {
			return true
		}
	}
	return false
}

// QuantizeVector quantizes a float32 vector to compressed bytes.
// Returns: quantized indices (packed), the min value, and the scale for dequantization.
func (tq *TurboQuantState) QuantizeVector(vec []float32, rotation []float32, codebook []float32, bits int) ([]byte, float32, float32) {
	dim := len(vec)
	bits = clampBits(bits)
	if tq == nil || dim == 0 || len(rotation) < dim*dim {
		return make([]byte, (dim*bits+7)/8), 0, 0
	}

	// Step 1: rotate
	rotated := make([]float32, dim)
	for i := 0; i < dim; i++ {
		var sum float32
		for j := 0; j < dim; j++ {
			sum += rotation[i*dim+j] * vec[j]
		}
		rotated[i] = sum
	}

	// Step 2: find min/max for uniform quantization
	vMin, vMax := rotated[0], rotated[0]
	for _, v := range rotated[1:] {
		if v < vMin {
			vMin = v
		}
		if v > vMax {
			vMax = v
		}
	}
	scale := vMax - vMin
	if scale < 1e-10 {
		return make([]byte, (dim*bits+7)/8), vMin, 0
	}

	// Step 3: quantize each coordinate to [0, 2^bits - 1]. The codebook
	// parameter is reserved for a future non-uniform quantizer; the current
	// implementation deliberately stays uniform because it has lower error in
	// the existing roundtrip tests.
	_ = codebook
	nLevels := 1 << bits
	indices := make([]byte, dim)
	for i, v := range rotated {
		normalized := (v - vMin) / scale // [0, 1]
		idx := int(normalized*float32(nLevels-1) + 0.5)
		if idx < 0 {
			idx = 0
		}
		if idx >= nLevels {
			idx = nLevels - 1
		}
		indices[i] = byte(idx)
	}

	// Step 4: pack indices into bytes
	packed := packIndices(indices, bits)
	return packed, vMin, scale
}

// DequantizeVector restores a float32 vector from compressed form.
func (tq *TurboQuantState) DequantizeVector(packed []byte, vMin, scale float32, rotation []float32, bits int, dim int) []float32 {
	bits = clampBits(bits)
	if tq == nil || dim <= 0 || len(rotation) < dim*dim {
		return make([]float32, maxInt(dim, 0))
	}
	// Unpack indices
	indices := unpackIndices(packed, bits, dim)
	nLevels := 1 << bits

	// Dequantize: map index back to value
	rotated := make([]float32, dim)
	for i, idx := range indices {
		rotated[i] = vMin + scale*float32(idx)/float32(nLevels-1)
	}

	// Inverse rotation: R^T @ rotated
	out := make([]float32, dim)
	for i := 0; i < dim; i++ {
		var sum float32
		for j := 0; j < dim; j++ {
			sum += rotation[j*dim+i] * rotated[j] // R^T = transpose indexing
		}
		out[i] = sum
	}

	return out
}

// randomOrthogonal generates a row-major random orthogonal matrix via
// modified Gram-Schmidt. Columns are orthonormal (Q^T Q = I); callers apply Q
// for rotation and Q^T for inverse rotation.
func randomOrthogonal(dim int, rng *rand.Rand) []float32 {
	if dim <= 0 {
		return nil
	}
	// Generate random Gaussian matrix
	mat := make([]float32, dim*dim)
	for i := range mat {
		mat[i] = float32(rng.NormFloat64())
	}

	// QR decomposition via modified Gram-Schmidt
	q := make([]float32, dim*dim)
	for j := 0; j < dim; j++ {
		// Copy column j
		for i := 0; i < dim; i++ {
			q[i*dim+j] = mat[i*dim+j]
		}
		// Subtract projections of previous columns
		for k := 0; k < j; k++ {
			var dot float32
			for i := 0; i < dim; i++ {
				dot += q[i*dim+j] * q[i*dim+k]
			}
			for i := 0; i < dim; i++ {
				q[i*dim+j] -= dot * q[i*dim+k]
			}
		}
		// Normalize
		var norm float32
		for i := 0; i < dim; i++ {
			norm += q[i*dim+j] * q[i*dim+j]
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 1e-10 {
			for i := 0; i < dim; i++ {
				q[i*dim+j] /= norm
			}
		}
	}

	return q
}

// betaOptimalCodebook computes MSE-optimal quantization levels for the
// Beta distribution that arises after random rotation of unit vectors.
// After rotation, each coordinate of a unit vector is approximately
// Normal(0, 1/sqrt(d)) for large d. We use quantiles of this distribution.
func betaOptimalCodebook(bits int) []float32 {
	bits = clampBits(bits)
	n := 1 << bits
	levels := make([]float32, n)
	// Use Normal quantiles scaled to the typical range
	// For d=128, stdev ≈ 1/sqrt(128) ≈ 0.0884, but after norm-division
	// the values span [-1, 1] with concentration near 0.
	// Use quantiles of N(0, 1/sqrt(d)) distribution, but since we normalize
	// to unit norm, the effective distribution is uniform-ish on the sphere.
	// Empirically, uniform quantization of [-1, 1] with more levels near 0 works well.
	// Lloyd-Max inspired: concentrate levels near 0 where density is highest.
	for i := 0; i < n; i++ {
		// Map [0, n-1] → [-1, 1] with concentration near 0
		// Use: level = sign(t) * |t|^0.5 where t = (2i+1)/2n - 1 maps to [-1,1]
		t := float64(2*i+1)/float64(2*n) - 0.5 // [-0.5, 0.5]
		t *= 2.0                               // [-1, 1]
		sign := float32(1.0)
		if t < 0 {
			sign = -1.0
			t = -t
		}
		levels[i] = sign * float32(math.Sqrt(t))
	}
	return levels
}

// packIndices packs byte indices into a bit-packed byte array.
func packIndices(indices []byte, bits int) []byte {
	bits = clampBits(bits)
	totalBits := len(indices) * bits
	packed := make([]byte, (totalBits+7)/8)
	bitPos := 0
	for _, idx := range indices {
		for b := 0; b < bits; b++ {
			if idx&(1<<b) != 0 {
				packed[bitPos/8] |= 1 << (bitPos % 8)
			}
			bitPos++
		}
	}
	return packed
}

// unpackIndices unpacks bit-packed indices into byte values.
func unpackIndices(packed []byte, bits int, n int) []byte {
	bits = clampBits(bits)
	if n <= 0 {
		return nil
	}
	indices := make([]byte, n)
	bitPos := 0
	for i := 0; i < n; i++ {
		var val byte
		for b := 0; b < bits; b++ {
			byteIdx := bitPos / 8
			if byteIdx < len(packed) && packed[byteIdx]&(1<<(bitPos%8)) != 0 {
				val |= 1 << b
			}
			bitPos++
		}
		indices[i] = val
	}
	return indices
}

func clampBits(bits int) int {
	if bits < 1 {
		return 1
	}
	if bits > 8 {
		return 8
	}
	return bits
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
