package simd

import (
	"math"
	"testing"
)

func TestVecAdd(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
	b := []float32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170}
	dst := make([]float32, len(a))
	VecAdd(dst, a, b)
	for i := range a {
		want := a[i] + b[i]
		if dst[i] != want {
			t.Fatalf("VecAdd[%d]=%f want %f", i, dst[i], want)
		}
	}
	t.Logf("VecAdd: OK (%d elements, HasVecAsm=%v)", len(a), HasVecAsm)
}

func TestVecMul(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	b := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	dst := make([]float32, len(a))
	VecMul(dst, a, b)
	for i := range a {
		want := a[i] * b[i]
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Fatalf("VecMul[%d]=%f want %f", i, dst[i], want)
		}
	}
	t.Log("VecMul: OK")
}

func TestSnrm2(t *testing.T) {
	x := []float32{3, 4} // sqrt(9+16) = 5
	got := Snrm2(x)
	if math.Abs(float64(got)-5.0) > 1e-5 {
		t.Fatalf("Snrm2([3,4])=%f want 5.0", got)
	}

	// Larger test
	x2 := make([]float32, 1024)
	for i := range x2 { x2[i] = 1.0 }
	got2 := Snrm2(x2)
	want2 := float32(math.Sqrt(1024))
	if math.Abs(float64(got2-want2)) > 0.01 {
		t.Fatalf("Snrm2(ones[1024])=%f want %f", got2, want2)
	}
	t.Logf("Snrm2: OK (%.4f)", got2)
}

func TestRMSNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	w := []float32{1, 1, 1, 1, 1, 1, 1, 1}
	eps := float32(1e-6)

	// Reference: rms = sqrt(mean(x^2))
	ss := float32(0)
	for _, v := range x { ss += v * v }
	rms := float32(math.Sqrt(float64(ss/8.0 + eps)))

	want := make([]float32, 8)
	for i := range x { want[i] = x[i] / rms }

	RMSNorm(x, w, eps)
	for i := range x {
		if math.Abs(float64(x[i]-want[i])) > 1e-5 {
			t.Fatalf("RMSNorm[%d]=%f want %f", i, x[i], want[i])
		}
	}

	// Test with non-trivial weights
	x2 := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
	w2 := make([]float32, len(x2))
	for i := range w2 { w2[i] = float32(i+1) * 0.1 }
	x2copy := make([]float32, len(x2))
	copy(x2copy, x2)

	RMSNorm(x2, w2, eps)

	// Verify against Go reference
	ss2 := float32(0)
	for _, v := range x2copy { ss2 += v * v }
	inv := float32(1.0 / math.Sqrt(float64(ss2/float32(len(x2copy))+eps)))
	for i := range x2 {
		want := w2[i] * x2copy[i] * inv
		if math.Abs(float64(x2[i]-want)) > 1e-4 {
			t.Fatalf("RMSNorm_weighted[%d]=%f want %f", i, x2[i], want)
		}
	}
	t.Logf("RMSNorm: OK (%d elements)", len(x2))
}

func TestToBF16(t *testing.T) {
	x := []float32{1.234567, -0.00123, 3.14159, 1000.5, 0}
	ToBF16(x)
	for i, v := range x {
		bits := math.Float32bits(v)
		if bits&0xFFFF != 0 {
			t.Fatalf("ToBF16[%d] lower bits not zeroed: %08x", i, bits)
		}
	}
	// 1.234567 → BF16 → ~1.234375
	if math.Abs(float64(x[0])-1.234375) > 0.01 {
		t.Fatalf("ToBF16(1.234567)=%f want ~1.234375", x[0])
	}
	t.Logf("ToBF16: OK (%.6f → %.6f)", 1.234567, x[0])
}

func TestVecSiLUMul(t *testing.T) {
	a := []float32{0, 1, -1, 2, -2, 0.5, 3, -0.5}
	b := []float32{1, 1, 1, 1, 1, 2, 0.5, 3}
	dst := make([]float32, len(a))
	VecSiLUMul(dst, a, b)
	for i := range a {
		x := a[i]
		silu := x / (1.0 + float32(math.Exp(float64(-x))))
		want := silu * b[i]
		if math.Abs(float64(dst[i]-want)) > 1e-5 {
			t.Fatalf("VecSiLUMul[%d]=%f want %f", i, dst[i], want)
		}
	}
	t.Log("VecSiLUMul: OK")
}

func BenchmarkRMSNorm(b *testing.B) {
	x := make([]float32, 3584)
	w := make([]float32, 3584)
	for i := range x { x[i] = float32(i)*0.001 - 1.0; w[i] = 1.0 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RMSNorm(x, w, 1e-6)
	}
}

func BenchmarkVecAdd(b *testing.B) {
	a := make([]float32, 3584)
	c := make([]float32, 3584)
	d := make([]float32, 3584)
	for i := range a { a[i] = float32(i); c[i] = float32(i) * 0.5 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VecAdd(d, a, c)
	}
}

func BenchmarkToBF16(b *testing.B) {
	x := make([]float32, 3584)
	for i := range x { x[i] = float32(i) * 0.001 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ToBF16(x)
	}
}
