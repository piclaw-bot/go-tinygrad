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
	if validBF16Buffer(&Buffer{Ptr: 1, Size: maxInt}, int(^uint32(0))+1) {
		t.Fatal("BF16 length exceeding CUDA u32 interface accepted")
	}
	if DevBF16RMSNorm(nil, nil, 1, 1e-6) {
		t.Fatal("nil BF16 RMSNorm reported success")
	}
	if DevBF16RMSNormNoScale(nil, 1, 1e-6) {
		t.Fatal("nil BF16 RMSNormNoScale reported success")
	}
	if DevBF16VecAdd(nil, nil, nil, 1) {
		t.Fatal("nil BF16 VecAdd reported success")
	}
	if DevBF16SiLUMul(nil, nil, nil, 1) {
		t.Fatal("nil BF16 SiLUMul reported success")
	}
	if DevBF16GELUTanhMul(nil, nil, 1) {
		t.Fatal("nil BF16 GELUTanhMul reported success")
	}
	if DevNativeBF16RMSNorm(nil, nil, 1, 1e-6) {
		t.Fatal("nil native BF16 RMSNorm reported success")
	}
	if DevNativeBF16VecAdd(nil, nil, nil, 1) {
		t.Fatal("nil native BF16 VecAdd reported success")
	}
}
