package quant

import (
	"strings"
	"testing"
)

func validGPTQInputs() ([]int32, []int32, []int32, []float32, int, int) {
	inFeatures, outFeatures := 8, 8
	qweight := make([]int32, (inFeatures/8)*outFeatures)
	qzeros := make([]int32, 1*(outFeatures/8))
	gIdx := make([]int32, inFeatures)
	scales := make([]float32, 1*outFeatures)
	return qweight, qzeros, gIdx, scales, inFeatures, outFeatures
}

func TestValidateGPTQAcceptsValidInputs(t *testing.T) {
	qw, qz, g, s, in, out := validGPTQInputs()
	if err := ValidateGPTQ(qw, qz, g, s, in, out, false); err != nil {
		t.Fatalf("ValidateGPTQ: %v", err)
	}
	if err := ValidateGPTQSym(qw, g, s, in, out); err != nil {
		t.Fatalf("ValidateGPTQSym: %v", err)
	}
	if got := DequantGPTQ(qw, qz, g, s, in, out, false); len(got) != in*out {
		t.Fatalf("DequantGPTQ len=%d, want %d", len(got), in*out)
	}
	if got := DequantGPTQSym(qw, g, s, in, out); len(got) != in*out {
		t.Fatalf("DequantGPTQSym len=%d, want %d", len(got), in*out)
	}
}

func TestValidateGPTQRejectsMalformedInputs(t *testing.T) {
	qw, qz, g, s, in, out := validGPTQInputs()
	cases := []struct {
		name string
		fn   func() error
		want string
	}{
		{name: "bad in", fn: func() error { return ValidateGPTQ(qw, qz, g, s, 7, out, false) }, want: "divisible"},
		{name: "bad out", fn: func() error { return ValidateGPTQ(qw, qz, g, s, in, 7, false) }, want: "outFeatures"},
		{name: "short qweight", fn: func() error { return ValidateGPTQ(qw[:1], qz, g, s, 16, out, false) }, want: "qweight"},
		{name: "short gidx", fn: func() error { return ValidateGPTQ(qw, qz, g[:1], s, in, out, false) }, want: "g_idx"},
		{name: "negative group", fn: func() error {
			bad := append([]int32(nil), g...)
			bad[0] = -1
			return ValidateGPTQ(qw, qz, bad, s, in, out, false)
		}, want: "negative"},
		{name: "short scales", fn: func() error {
			bad := append([]int32(nil), g...)
			bad[0] = 1
			return ValidateGPTQ(qw, qz, bad, s, in, out, false)
		}, want: "scales"},
		{name: "short qzeros", fn: func() error {
			bad := append([]int32(nil), g...)
			bad[0] = 1
			ss := make([]float32, 16)
			return ValidateGPTQ(qw, qz, bad, ss, in, out, false)
		}, want: "qzeros"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.fn()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestDequantGPTQRejectsMalformedInputsWithoutPanic(t *testing.T) {
	qw, qz, g, s, _, out := validGPTQInputs()
	if got := DequantGPTQ(qw[:1], qz, g, s, 16, out, false); got != nil {
		t.Fatalf("DequantGPTQ malformed len=%d, want nil", len(got))
	}
	if got := DequantGPTQSym(qw[:1], g, s, 16, out); got != nil {
		t.Fatalf("DequantGPTQSym malformed len=%d, want nil", len(got))
	}
}
