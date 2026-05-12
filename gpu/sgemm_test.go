package gpu

import "testing"

func TestSgemmValidationRejectsMalformedInputs(t *testing.T) {
	if err := Sgemm(1, 1, 1, 1, nil, nil, nil); err == nil {
		t.Fatal("Sgemm accepted nil buffers")
	}
	if _, err := SgemmHost(0, 1, 1, 1, nil, nil); err == nil {
		t.Fatal("SgemmHost accepted zero M")
	}
	if _, err := SgemmHost(2, 2, 2, 1, []float32{1}, make([]float32, 4)); err == nil {
		t.Fatal("SgemmHost accepted short A")
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := SgemmHost(maxInt/2+1, 3, 2, 1, nil, nil); err == nil {
		t.Fatal("SgemmHost accepted overflowing dimensions")
	}
	if err := Sgemm(2, 2, 2, 1, &Buffer{Ptr: 1, Size: 4}, &Buffer{Ptr: 1, Size: 16}, &Buffer{Ptr: 1, Size: 16}); err == nil {
		t.Fatal("Sgemm accepted short A buffer")
	}
}
