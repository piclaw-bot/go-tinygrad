package quant

import (
	"strings"
	"testing"
)

func validGemvQ4Inputs() ([]float32, []float32, []int32, []int32, []float32, int, int) {
	inDim, outDim := 8, 2
	out := make([]float32, outDim)
	x := make([]float32, inDim)
	qweight := make([]int32, (inDim/8)*outDim)
	gIdx := make([]int32, inDim)
	scales := make([]float32, outDim)
	return out, x, qweight, gIdx, scales, inDim, outDim
}

func TestValidateGemvQ4Sym(t *testing.T) {
	out, x, qw, g, s, inDim, outDim := validGemvQ4Inputs()
	if err := ValidateGemvQ4Sym(out, x, qw, g, s, inDim, outDim); err != nil {
		t.Fatalf("ValidateGemvQ4Sym: %v", err)
	}
	cases := []struct {
		name string
		fn   func() error
		want string
	}{
		{name: "short out", fn: func() error { return ValidateGemvQ4Sym(out[:1], x, qw, g, s, inDim, outDim) }, want: "out length"},
		{name: "short x", fn: func() error { return ValidateGemvQ4Sym(out, x[:1], qw, g, s, inDim, outDim) }, want: "x length"},
		{name: "short qweight", fn: func() error { return ValidateGemvQ4Sym(out, x, nil, g, s, inDim, outDim) }, want: "qweight"},
		{name: "bad group", fn: func() error {
			bad := append([]int32(nil), g...)
			bad[0] = 1
			return ValidateGemvQ4Sym(out, x, qw, bad, s, inDim, outDim)
		}, want: "scales"},
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

func TestGemvQ4SymMalformedDoesNotPanic(t *testing.T) {
	out, x, qw, g, s, inDim, outDim := validGemvQ4Inputs()
	out[0] = 123
	GemvQ4Sym(out, x, qw[:1], g, s, 16, outDim)
	if out[0] != 123 {
		t.Fatalf("malformed GemvQ4Sym should leave output unchanged, got %f", out[0])
	}
	GemvQ4Sym(out[:1], x, qw, g, s, inDim, outDim)
}
