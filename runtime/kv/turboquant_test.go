package kv

import (
	"math"
	"testing"
)

func TestTurboQuantRoundtrip(t *testing.T) {
	dim := 128
	tq := NewTurboQuantState(dim, 28, DefaultTurboQuantConfig())

	// Create a test vector
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(math.Sin(float64(i)*0.1)) * 2.0
	}

	// Quantize keys (4-bit)
	packed, vMin, scale := tq.QuantizeVector(vec, tq.RotationK, tq.CodebookK, tq.Config.KeyBits)
	restored := tq.DequantizeVector(packed, vMin, scale, tq.RotationK, tq.Config.KeyBits, dim)

	// Measure reconstruction error
	var maxErr, sumErr float64
	for i := range vec {
		d := math.Abs(float64(vec[i] - restored[i]))
		sumErr += d
		if d > maxErr {
			maxErr = d
		}
	}
	meanErr := sumErr / float64(dim)
	t.Logf("4-bit key roundtrip: maxErr=%.6f meanErr=%.6f", maxErr, meanErr)
	t.Logf("  original[0:5]: %v", vec[:5])
	t.Logf("  restored[0:5]: %v", restored[:5])

	// 4-bit should have reasonable error (< 0.5 for values in [-2, 2])
	if maxErr > 1.0 {
		t.Errorf("4-bit key error too large: maxErr=%.6f", maxErr)
	}

	// Quantize values (2-bit)
	packed2, vMin2, scale2 := tq.QuantizeVector(vec, tq.RotationV, tq.CodebookV, tq.Config.ValueBits)
	restored2 := tq.DequantizeVector(packed2, vMin2, scale2, tq.RotationV, tq.Config.ValueBits, dim)

	var maxErr2, sumErr2 float64
	for i := range vec {
		d := math.Abs(float64(vec[i] - restored2[i]))
		sumErr2 += d
		if d > maxErr2 {
			maxErr2 = d
		}
	}
	meanErr2 := sumErr2 / float64(dim)
	t.Logf("2-bit value roundtrip: maxErr=%.6f meanErr=%.6f", maxErr2, meanErr2)

	// 2-bit should have larger but bounded error
	if maxErr2 > 3.0 {
		t.Errorf("2-bit value error too large: maxErr=%.6f", maxErr2)
	}
}

func TestTurboQuantCompressionRatio(t *testing.T) {
	dim := 128

	// Original: 128 × 4 bytes = 512 bytes per vector
	origBytes := dim * 4

	// 4-bit: 128 × 4 bits / 8 = 64 bytes + 4 bytes norm = 68 bytes
	bits4Bytes := (dim*4+7)/8 + 4
	// 2-bit: 128 × 2 bits / 8 = 32 bytes + 4 bytes norm = 36 bytes
	bits2Bytes := (dim*2+7)/8 + 4

	t.Logf("Compression ratios for dim=%d:", dim)
	t.Logf("  Original: %d bytes", origBytes)
	t.Logf("  4-bit key: %d bytes (%.1f×)", bits4Bytes, float64(origBytes)/float64(bits4Bytes))
	t.Logf("  2-bit val: %d bytes (%.1f×)", bits2Bytes, float64(origBytes)/float64(bits2Bytes))
	t.Logf("  K+V pair: %d bytes (%.1f× vs %d)", bits4Bytes+bits2Bytes,
		float64(2*origBytes)/float64(bits4Bytes+bits2Bytes), 2*origBytes)
}

func TestTurboQuantProtectedLayers(t *testing.T) {
	var nilTQ *TurboQuantState
	if nilTQ.IsProtectedLayer(0) {
		t.Fatal("nil TurboQuantState reported protected layer")
	}
	tq := NewTurboQuantState(128, 28, DefaultTurboQuantConfig())

	// First 2 and last 2 should be protected
	for _, l := range []int{0, 1, 26, 27} {
		if !tq.IsProtectedLayer(l) {
			t.Errorf("layer %d should be protected", l)
		}
	}
	for _, l := range []int{-1, 5, 14, 20} {
		if tq.IsProtectedLayer(l) {
			t.Errorf("layer %d should NOT be protected", l)
		}
	}
}

func TestTurboQuantOrthogonality(t *testing.T) {
	dim := 64
	tq := NewTurboQuantState(dim, 28, DefaultTurboQuantConfig())

	// Check R @ R^T ≈ I
	maxOffDiag := float64(0)
	minOnDiag := float64(2)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var dot float64
			for k := 0; k < dim; k++ {
				dot += float64(tq.RotationK[i*dim+k]) * float64(tq.RotationK[j*dim+k])
			}
			if i == j {
				if dot < minOnDiag {
					minOnDiag = dot
				}
			} else {
				if math.Abs(dot) > maxOffDiag {
					maxOffDiag = math.Abs(dot)
				}
			}
		}
	}
	t.Logf("Orthogonality check: minOnDiag=%.6f maxOffDiag=%.6g", minOnDiag, maxOffDiag)
	if maxOffDiag > 1e-5 || minOnDiag < 0.999 {
		t.Errorf("rotation matrix not orthogonal: minOnDiag=%.6f maxOffDiag=%.6g", minOnDiag, maxOffDiag)
	}
}

func TestCompressedKVCacheMalformedLayoutGuards(t *testing.T) {
	tq := NewTurboQuantState(4, 1, TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 0})
	bad := NewCompressedKVCache(4, 3, 2, tq, false)
	bad.Append([]float32{1, 2, 3, 4}, []float32{5, 6, 7, 8})
	if bad.CompressedCount() != 0 {
		t.Fatalf("inconsistent head layout compressed entries: %d", bad.CompressedCount())
	}

	c := NewCompressedKVCache(4, 1, 4, tq, false)
	c.FullK = []float32{1, 2, 3, 4, 9, 9, 9, 9}
	c.FullV = []float32{5, 6, 7, 8, 9, 9, 9, 9}
	c.seqLen = 1
	if got := c.GetK(); len(got) != 4 {
		t.Fatalf("GetK did not clamp full cache to seqLen: len=%d", len(got))
	}
	c.CompressedK = []compressedEntry{{Packed: []byte{1}, HeadVMin: nil, HeadScale: nil}}
	c.CompressedV = []compressedEntry{{Packed: []byte{1}, HeadVMin: nil, HeadScale: nil}}
	if got := c.GetK(); len(got) != len(c.FullK) {
		t.Fatalf("malformed compressed K should fall back to FullK, len=%d", len(got))
	}
	if got := c.GetV(); len(got) != len(c.FullV) {
		t.Fatalf("malformed compressed V should fall back to FullV, len=%d", len(got))
	}
}

func TestTurboQuantOverflowGuards(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tooLarge := maxInt/2 + 1
	if got := squareSize(tooLarge); got != -1 {
		t.Fatalf("squareSize overflow=%d, want -1", got)
	}
	if got := packedByteLen(maxInt, 8); got != -1 {
		t.Fatalf("packedByteLen overflow=%d, want -1", got)
	}
	if got := randomOrthogonal(tooLarge, nil); got != nil {
		t.Fatalf("randomOrthogonal malformed returned len=%d", len(got))
	}
	state := NewTurboQuantState(tooLarge, 1, DefaultTurboQuantConfig())
	if state.HeadDim != 0 || len(state.RotationK) != 0 || len(state.RotationV) != 0 {
		t.Fatalf("overflowing headDim not sanitized: headDim=%d rk=%d rv=%d", state.HeadDim, len(state.RotationK), len(state.RotationV))
	}
	packed := packIndices(make([]byte, 4), 4)
	if len(packed) != 2 {
		t.Fatalf("packIndices valid len=%d want 2", len(packed))
	}
}
