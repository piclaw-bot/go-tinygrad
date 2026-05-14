package gpu

import (
	"math"
	"testing"
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
