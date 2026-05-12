package model

import "testing"

func TestGemvHelpersMalformedInputsDoNotPanic(t *testing.T) {
	out := []float32{1, 2}
	gemv(out, nil, nil, 4, 2)
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("gemv malformed did not zero output: %v", out)
	}
	out = []float32{1, 2}
	gemvNT(out, []float32{1}, []float32{1}, 4, 2)
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("gemvNT malformed did not zero output: %v", out)
	}
	out = []float32{1, 2}
	gemvNTParallel(out, []float32{1}, []float32{1}, 4, 2)
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("gemvNTParallel malformed did not zero output: %v", out)
	}
}

func TestGemvHelpersValid(t *testing.T) {
	x := []float32{2, 3}
	// gemv expects pre-transposed [inDim,outDim].
	wNN := []float32{1, 10, 2, 20}
	out := make([]float32, 2)
	gemv(out, x, wNN, 2, 2)
	if out[0] != 8 || out[1] != 80 {
		t.Fatalf("gemv=%v want [8 80]", out)
	}
	// gemvNT expects [outDim,inDim].
	wNT := []float32{1, 2, 10, 20}
	gemvNT(out, x, wNT, 2, 2)
	if out[0] != 8 || out[1] != 80 {
		t.Fatalf("gemvNT=%v want [8 80]", out)
	}
	gemvNTParallel(out, x, wNT, 2, 2)
	if out[0] != 8 || out[1] != 80 {
		t.Fatalf("gemvNTParallel=%v want [8 80]", out)
	}
}

func TestLowLevelHelpersRejectOverflowProducts(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	out := []float32{1, 2}
	gemv(out, []float32{1, 2}, []float32{1, 2}, maxInt/2+1, 3)
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("gemv overflow did not zero output: %v", out)
	}
	out = []float32{1, 2}
	gemvNT(out, []float32{1, 2}, []float32{1, 2}, maxInt/2+1, 3)
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("gemvNT overflow did not zero output: %v", out)
	}
	out = []float32{1, 2}
	gemvNTParallel(out, []float32{1, 2}, []float32{1, 2}, maxInt/2+1, 3)
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("gemvNTParallel overflow did not zero output: %v", out)
	}
	attnOut := []float32{9, 9}
	gqaAttentionScaleInto(attnOut, make([]float32, 1), make([]float32, 1), make([]float32, 1), make([]float32, 1), maxInt/2+1, 1, 1, 3, 1)
	if attnOut[0] != 9 || attnOut[1] != 9 {
		t.Fatalf("attention overflow mutated output: %v", attnOut)
	}
}
