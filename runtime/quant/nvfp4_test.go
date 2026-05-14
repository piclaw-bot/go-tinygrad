package quant

import "testing"

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
