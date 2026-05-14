package quant

import (
	"math"
	"testing"
)

func TestDecodeFP4E2M1Codebook(t *testing.T) {
	want := []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}
	for code, w := range want {
		got := DecodeFP4E2M1(byte(code))
		if got != w {
			t.Fatalf("DecodeFP4E2M1(%#x)=%v want %v", code, got, w)
		}
	}
}

func TestDecodeF8E4M3(t *testing.T) {
	cases := []struct {
		code byte
		want float32
	}{
		{0x00, 0},
		{0x01, 1.0 / 512},
		{0x08, 1.0 / 64},
		{0x38, 1},
		{0x3c, 1.5},
		{0x40, 2},
		{0xb8, -1},
	}
	for _, tc := range cases {
		got := DecodeF8E4M3(tc.code)
		if math.Abs(float64(got-tc.want)) > 1e-7 {
			t.Fatalf("DecodeF8E4M3(%#x)=%v want %v", tc.code, got, tc.want)
		}
	}
}

func TestUnpackNVFP4LowNibbleFirst(t *testing.T) {
	got := UnpackNVFP4([]byte{0x10, 0x32, 0xba}, 6)
	want := []float32{0, 0.5, 1, 1.5, -1, -1.5}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestDequantNVFP4Synthetic(t *testing.T) {
	qw := syntheticNVFP4Weight()
	got := DequantNVFP4(qw)
	want := []float32{0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 0, -0.25, -0.5, -0.75, -1, -1.5, -2, -3}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestGemvNVFP4MatchesDequantizedReference(t *testing.T) {
	qw := syntheticNVFP4Weight()
	x := []float32{1, -1, 2, -2, 0.5, -0.5, 3, -3, 1, 1, 1, 1, -1, -1, -1, -1}
	wantWeights := DequantNVFP4(qw)
	want := float32(0)
	for i, w := range wantWeights {
		want += w * x[i]
	}
	got := []float32{123}
	GemvNVFP4(got, x, qw)
	if math.Abs(float64(got[0]-want)) > 1e-6 {
		t.Fatalf("GemvNVFP4=%v want %v", got[0], want)
	}
}

func TestNVFP4TinySyntheticLogitsMatchF32Reference(t *testing.T) {
	qw := syntheticNVFP4LogitWeight()
	hidden := []float32{0.25, -0.5, 1.5, -2, 0.75, -1.25, 2.5, -3, 1, 0.5, -0.75, 1.25, -1.5, 2, -2.5, 3}

	got := make([]float32, qw.OutDim)
	GemvNVFP4(got, hidden, qw)

	weights := DequantNVFP4(qw)
	want := make([]float32, qw.OutDim)
	for row := 0; row < qw.OutDim; row++ {
		for col := 0; col < qw.InDim; col++ {
			want[row] += weights[row*qw.InDim+col] * hidden[col]
		}
	}

	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("logit[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func syntheticNVFP4Weight() *NVFP4Weight {
	return &NVFP4Weight{
		// Low nibble first: codes 0..15 across one 16-value group.
		Weight:       []byte{0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe},
		WeightScale:  []byte{0x40}, // E4M3 2.0
		WeightScale2: 0.25,
		OutDim:       1,
		InDim:        16,
		Groups:       1,
		GroupSize:    16,
	}
}

func syntheticNVFP4LogitWeight() *NVFP4Weight {
	return &NVFP4Weight{
		// Three vocab rows, each one 16-value group. The bytes deliberately mix
		// positive/negative FP4 codes so logits exercise signs and scale handling.
		Weight: []byte{
			0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
			0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
			0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f,
		},
		WeightScale: []byte{
			0x38, // 1.0
			0x40, // 2.0
			0x34, // 0.75
		},
		WeightScale2: 0.5,
		OutDim:       3,
		InDim:        16,
		Groups:       1,
		GroupSize:    16,
	}
}

func TestValidateNVFP4WeightObservedLayouts(t *testing.T) {
	cases := []struct {
		name      string
		outDim    int
		packedIn  int
		scaleCols int
	}{
		{"qwen3 dense q_proj", 4096, 2048, 256},
		{"qwen3 dense down_proj", 4096, 6144, 768},
		{"qwen3 moe expert down_proj", 2048, 384, 48},
		{"gemma4 dense down_proj", 5376, 10752, 1344},
		{"gemma4 moe expert scale", 2816, 352, 44},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			qw := &NVFP4Weight{
				Weight:      make([]byte, tc.outDim*tc.packedIn),
				WeightScale: make([]byte, tc.outDim*tc.scaleCols),
				OutDim:      tc.outDim,
				InDim:       tc.packedIn * 2,
				Groups:      tc.scaleCols,
				GroupSize:   16,
			}
			if err := ValidateNVFP4Weight(qw); err != nil {
				t.Fatalf("ValidateNVFP4Weight: %v", err)
			}
		})
	}
}

func TestValidateNVFP4WeightRejectsBadLayout(t *testing.T) {
	cases := []struct {
		name string
		qw   *NVFP4Weight
	}{
		{"nil", nil},
		{"odd input", &NVFP4Weight{Weight: make([]byte, 1), WeightScale: make([]byte, 1), OutDim: 1, InDim: 3, Groups: 1, GroupSize: 3}},
		{"bad groups", &NVFP4Weight{Weight: make([]byte, 8), WeightScale: make([]byte, 2), OutDim: 2, InDim: 8, Groups: 2, GroupSize: 8}},
		{"short weight", &NVFP4Weight{Weight: make([]byte, 7), WeightScale: make([]byte, 2), OutDim: 2, InDim: 8, Groups: 1, GroupSize: 8}},
		{"short scale", &NVFP4Weight{Weight: make([]byte, 8), WeightScale: make([]byte, 1), OutDim: 2, InDim: 8, Groups: 1, GroupSize: 8}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateNVFP4Weight(tc.qw); err == nil {
				t.Fatal("ValidateNVFP4Weight succeeded, want error")
			}
		})
	}
}
