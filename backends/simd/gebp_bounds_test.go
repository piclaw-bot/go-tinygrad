package simd

import (
	"testing"
	"unsafe"
)

func TestGEBPValidationRejectsMalformedArgs(t *testing.T) {
	if ensureGebpBuf(-1) != nil || ensureGebpBuf(0) != nil {
		t.Fatal("ensureGebpBuf should return nil for non-positive sizes")
	}
	if validGEBPArgs(1, 1, 1, nil, unsafe.Pointer(&[]float32{1}[0]), unsafe.Pointer(&[]float32{1}[0]), 1, 1, 1) {
		t.Fatal("nil pointer args accepted")
	}
	a := []float32{1, 2}
	b := []float32{3, 4}
	c := []float32{0}
	if validGEBPArgs(1, 2, 2, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), 1, 2, 2) {
		t.Fatal("short lda accepted")
	}
	SgemmNTGebp(1, 1, 1, 1, nil, nil, nil, 1, 1, 1)
	bp := make([]float32, gebpNR)
	packBNTScalar([]float32{1}, 1, 0, 2, 1, bp)
	for i, v := range bp {
		if v != 0 {
			t.Fatalf("packBNTScalar malformed wrote bp[%d]=%f", i, v)
		}
	}
	maxInt := int(^uint(0) >> 1)
	if validPackBNTArgs(make([]float32, 1), maxInt, maxInt, 2, 2, make([]float32, gebpNR)) {
		t.Fatal("overflowing packBNT args accepted")
	}
}

func TestSgemmNTBlockedValidationRejectsMalformedArgs(t *testing.T) {
	SgemmNTBlockedFMA(1, 1, 1, 1, nil, nil, nil, 1, 1, 1)
	a := []float32{1}
	b := []float32{1}
	c := []float32{42}
	SgemmNTBlockedFMA(1, 1, 2, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), 1, 1, 1)
	if c[0] != 42 {
		t.Fatalf("malformed blocked SGEMM mutated C=%v", c[0])
	}
}

func TestSgemmNTGatherValidationRejectsMalformedArgs(t *testing.T) {
	SgemmNTGather(1, 1, 1, 1, nil, nil, nil, 1, 1, 1)
	a := []float32{1}
	b := []float32{1}
	c := []float32{42}
	SgemmNTGather(1, 1, 2, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), 1, 1, 1)
	if c[0] != 42 {
		t.Fatalf("malformed gather SGEMM mutated C=%v", c[0])
	}
	SgemmNTGather(1, 1, 1, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), 1, int(int32Max)/7+1, 1)
	if c[0] != 42 {
		t.Fatalf("overflowing gather index mutated C=%v", c[0])
	}
}
