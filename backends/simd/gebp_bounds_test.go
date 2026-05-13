package simd

import (
	"math"
	"sync"
	"testing"
	"unsafe"
)

func TestCheckedFloat32ByteOffsetRejectsOverflow(t *testing.T) {
	if off, ok := checkedFloat32ByteOffset(3); !ok || off != 12 {
		t.Fatalf("checkedFloat32ByteOffset(3)=(%d,%v), want (12,true)", off, ok)
	}
	maxInt := int(^uint(0) >> 1)
	if _, ok := checkedFloat32ByteOffset(maxInt/4 + 1); ok {
		t.Fatal("overflowing float32 byte offset accepted")
	}
}

func TestGEBPValidationRejectsMalformedArgs(t *testing.T) {
	if makeGebpBuf(-1) != nil || makeGebpBuf(0) != nil {
		t.Fatal("makeGebpBuf should return nil for non-positive sizes")
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

func TestMakeGebpBufReturnsIndependentScratch(t *testing.T) {
	a := makeGebpBuf(gebpNR)
	b := makeGebpBuf(gebpNR)
	if len(a) != gebpNR || len(b) != gebpNR {
		t.Fatalf("unexpected scratch lengths: %d %d", len(a), len(b))
	}
	a[0] = 123
	if b[0] != 0 {
		t.Fatalf("scratch buffers alias: b[0]=%f", b[0])
	}
}

func TestSgemmNTGebpConcurrentScratch(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("SGEMM assembly unavailable on this runtime")
	}
	const (
		m = 5
		n = 7
		k = 9
	)
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	for i := range a {
		a[i] = float32((i%11)-5) * 0.125
	}
	for i := range b {
		b[i] = float32((i%13)-6) * 0.0625
	}
	want := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[j*k+p]
			}
			want[i*n+j] = sum
		}
	}

	const workers = 32
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for iter := 0; iter < 64; iter++ {
				c := make([]float32, m*n)
				SgemmNTGebp(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), k, k, n)
				for idx, got := range c {
					if math.Abs(float64(got-want[idx])) > 1e-4 {
						t.Errorf("worker %d iter %d c[%d]=%f want %f", worker, iter, idx, got, want[idx])
						return
					}
				}
			}
		}(w)
	}
	wg.Wait()
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
