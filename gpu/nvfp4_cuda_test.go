package gpu

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/quant"
)

func TestSupportsNativeNVFP4TensorCore(t *testing.T) {
	cases := []struct {
		major int
		minor int
		want  bool
	}{
		{8, 0, false},
		{8, 9, false},
		{9, 0, false},
		{10, 0, true},
		{12, 0, true},
	}
	for _, tc := range cases {
		if got := supportsNativeNVFP4TensorCore(tc.major, tc.minor); got != tc.want {
			t.Fatalf("supportsNativeNVFP4TensorCore(%d,%d)=%v want %v", tc.major, tc.minor, got, tc.want)
		}
	}
}

func TestNVFP4RequiredBytes(t *testing.T) {
	weightBytes, scaleBytes, err := nvfp4RequiredBytes(4096, 4096, 256)
	if err != nil {
		t.Fatalf("nvfp4RequiredBytes: %v", err)
	}
	if weightBytes != 4096*2048 || scaleBytes != 4096*256 {
		t.Fatalf("weightBytes=%d scaleBytes=%d", weightBytes, scaleBytes)
	}
}

func TestNVFP4RequiredBytesRejectsBadDims(t *testing.T) {
	if _, _, err := nvfp4RequiredBytes(1, 3, 1); err == nil {
		t.Fatal("nvfp4RequiredBytes accepted odd inDim")
	}
}

func TestBytesAsFloat32PaddedRoundTripsRawBytes(t *testing.T) {
	input := []byte{0x01, 0x02, 0x03, 0x04, 0x05}
	got := bytesAsFloat32Padded(input)
	if len(got) != 2 {
		t.Fatalf("len=%d want 2", len(got))
	}
	if math.Float32bits(got[0]) != 0x04030201 {
		t.Fatalf("bits[0]=%#x", math.Float32bits(got[0]))
	}
	if math.Float32bits(got[1]) != 0x00000005 {
		t.Fatalf("bits[1]=%#x", math.Float32bits(got[1]))
	}
	if roundTrip := float32PackedAsBytes(got, len(input)); string(roundTrip) != string(input) {
		t.Fatalf("roundTrip=%#v want %#v", roundTrip, input)
	}
}

func TestGemvNVFP4RejectsInvalidBuffers(t *testing.T) {
	w := &GPUNVFP4Weight{
		Weight:      &Buffer{Size: f32SlotsForBytes(8) * 4},
		WeightScale: &Buffer{Size: f32SlotsForBytes(2) * 4},
		OutDim:      2,
		InDim:       8,
		Groups:      1,
		GroupSize:   8,
		WeightBytes: 8,
		ScaleBytes:  2,
	}
	if err := GemvNVFP4(make([]float32, 1), make([]float32, 8), w); err == nil {
		t.Fatal("GemvNVFP4 accepted short output")
	}
	if err := GemvNVFP4(make([]float32, 2), make([]float32, 7), w); err == nil {
		t.Fatal("GemvNVFP4 accepted short input")
	}
}

func TestGemvNVFP4F32RejectsOverflowShape(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if err := gemvNVFP4F32(make([]float32, 1), make([]float32, 1), maxInt/2+1, 3, make([]float32, 1)); err == nil {
		t.Fatal("gemvNVFP4F32 accepted overflowing shape")
	}
}

func TestGemvNVFP4F32WithReferenceDequant(t *testing.T) {
	qw := &quant.NVFP4Weight{
		Weight: []byte{
			0x10, 0x32, 0x54, 0x76,
			0x98, 0xba, 0xdc, 0xfe,
		},
		WeightScale:  []byte{0x38, 0x40}, // 1.0, 2.0
		WeightScale2: 0.5,
		OutDim:       2,
		InDim:        8,
		Groups:       1,
		GroupSize:    8,
	}
	weights := quant.DequantNVFP4(qw)
	x := []float32{1, -1, 2, -2, 0.5, -0.5, 3, -3}
	want := make([]float32, 2)
	for row := 0; row < qw.OutDim; row++ {
		for col := 0; col < qw.InDim; col++ {
			want[row] += weights[row*qw.InDim+col] * x[col]
		}
	}
	got := make([]float32, 2)
	if err := gemvNVFP4F32(got, x, qw.OutDim, qw.InDim, weights); err != nil {
		t.Fatalf("gemvNVFP4F32: %v", err)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestValidGPUNVFP4Weight(t *testing.T) {
	w := &GPUNVFP4Weight{
		Weight:      &Buffer{Size: f32SlotsForBytes(8) * 4},
		WeightScale: &Buffer{Size: f32SlotsForBytes(2) * 4},
		OutDim:      2,
		InDim:       8,
		Groups:      1,
		GroupSize:   8,
		WeightBytes: 8,
		ScaleBytes:  2,
	}
	if !validGPUNVFP4Weight(w) {
		t.Fatal("validGPUNVFP4Weight rejected valid synthetic layout")
	}
	w.ScaleBytes = 1
	if validGPUNVFP4Weight(w) {
		t.Fatal("validGPUNVFP4Weight accepted wrong scale byte count")
	}
}
