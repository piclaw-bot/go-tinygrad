package gpu

import "testing"

func TestBF16DispatchValidation(t *testing.T) {
	if validBF16Buffer(nil, 1) {
		t.Fatal("nil BF16 buffer accepted")
	}
	if validBF16Buffer(&Buffer{Ptr: 1, Size: 1}, 1) {
		t.Fatal("short BF16 buffer accepted")
	}
	if !validBF16Buffer(&Buffer{Ptr: 1, Size: 2}, 1) {
		t.Fatal("valid BF16 buffer rejected")
	}
	maxInt := int(^uint(0) >> 1)
	if validBF16Buffer(&Buffer{Ptr: 1, Size: maxInt}, maxInt/2+1) {
		t.Fatal("overflowing BF16 length accepted")
	}
	DevBF16RMSNorm(nil, nil, 1, 1e-6)
	DevBF16RMSNormNoScale(nil, 1, 1e-6)
	DevBF16VecAdd(nil, nil, nil, 1)
	DevBF16SiLUMul(nil, nil, nil, 1)
	DevBF16GELUTanhMul(nil, nil, 1)
	DevNativeBF16RMSNorm(nil, nil, 1, 1e-6)
	DevNativeBF16VecAdd(nil, nil, nil, 1)
}
