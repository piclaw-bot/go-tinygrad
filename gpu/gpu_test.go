package gpu

import (
	"testing"
)

func TestGPUAvailable(t *testing.T) {
	if !Available() {
		t.Skip("GPU not available")
	}
	t.Log("GPU available via cuBLAS")
}

func TestGPUSgemm(t *testing.T) {
	if !Available() {
		t.Skip("GPU not available")
	}

	// 2×3 @ 3×2 = 2×2
	m, k, n := 2, 3, 2
	A := Alloc(m * k)
	B := Alloc(k * n)
	C := Alloc(m * n)
	defer A.Free()
	defer B.Free()
	defer C.Free()

	aData := []float32{1, 2, 3, 4, 5, 6}
	bData := []float32{7, 8, 9, 10, 11, 12}
	A.Upload(aData)
	B.Upload(bData)

	SgemmNN(m, n, k, 1.0, A, B, C, k, n, n)
	Sync()

	cData := make([]float32, m*n)
	C.Download(cData)
	// [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	// [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	want := []float32{58, 64, 139, 154}
	for i, v := range cData {
		if v != want[i] {
			t.Fatalf("C[%d]=%v want %v", i, v, want[i])
		}
	}
	t.Logf("GPU SGEMM result: %v ✓", cData)
}
