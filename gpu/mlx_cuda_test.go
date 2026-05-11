package gpu

import "testing"

func TestUploadMLXWeightValidation(t *testing.T) {
	if _, err := UploadMLXWeight(nil, nil, nil, 0, 8, 64, false); err == nil {
		t.Fatalf("expected invalid dimension error")
	}
	if !SgemmReady() {
		t.Skip("GPU not available; dimension validation checked before readiness")
	}
	if _, err := UploadMLXWeight(make([]uint32, 1), make([]float32, 8), make([]float32, 8), 16, 8, 8, false); err == nil {
		t.Fatalf("expected short weight error")
	}
	if _, err := UploadMLXWeight(make([]uint32, 16), make([]float32, 1), make([]float32, 8), 16, 8, 8, false); err == nil {
		t.Fatalf("expected short scale error")
	}
}

func TestMLXDispatchMalformedDoesNotPanic(t *testing.T) {
	out := NewDevBuf(1)
	x := NewDevBufFrom([]float32{1})
	GemvMLX(out, x, nil)
	GemmMLX(out, x, nil, 2)
	GemvMLXDirect(out, x, nil)
}
