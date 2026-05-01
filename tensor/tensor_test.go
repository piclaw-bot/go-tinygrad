package tensor

import (
	"math"
	"testing"
)

func approx(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func TestFromFloat32(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	x := FromFloat32(data, []int{2, 3})
	if x.Numel() != 6 {
		t.Fatalf("numel=%d want 6", x.Numel())
	}
	got := x.Data()
	for i, v := range got {
		if v != data[i] {
			t.Fatalf("data[%d]=%v want %v", i, v, data[i])
		}
	}
}

func TestAdd(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3}, []int{3})
	b := FromFloat32([]float32{4, 5, 6}, []int{3})
	c := a.Add(b)
	got := c.Data()
	want := []float32{5, 7, 9}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("c[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestMul(t *testing.T) {
	a := FromFloat32([]float32{2, 3, 4}, []int{3})
	b := FromFloat32([]float32{5, 6, 7}, []int{3})
	c := a.Mul(b)
	got := c.Data()
	want := []float32{10, 18, 28}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("c[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestChainedOps(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3}, []int{3})
	b := FromFloat32([]float32{4, 5, 6}, []int{3})
	// (a + b) * a = [5, 14, 27]
	c := a.Add(b).Mul(a)
	got := c.Data()
	want := []float32{5, 14, 27}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("c[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestNeg(t *testing.T) {
	a := FromFloat32([]float32{1, -2, 3}, []int{3})
	got := a.Neg().Data()
	want := []float32{-1, 2, -3}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestSqrt(t *testing.T) {
	a := FromFloat32([]float32{1, 4, 9, 16}, []int{4})
	got := a.Sqrt().Data()
	want := []float32{1, 2, 3, 4}
	for i := range got {
		if !approx(got[i], want[i], 1e-6) {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestReduceSum(t *testing.T) {
	// 2×3 matrix, sum over axis 1
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	got := a.Sum(1).Data()
	// [1+2+3, 4+5+6] = [6, 15]
	want := []float32{6, 15}
	for i := range got {
		if !approx(got[i], want[i], 1e-5) {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestReduceMax(t *testing.T) {
	a := FromFloat32([]float32{1, 5, 3, 4, 2, 6}, []int{2, 3})
	got := a.ReduceMax(1).Data()
	want := []float32{5, 6}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestReshape(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := a.Reshape([]int{3, 2})
	if b.Shape()[0] != 3 || b.Shape()[1] != 2 {
		t.Fatalf("shape=%v want [3 2]", b.Shape())
	}
	got := b.Data()
	want := []float32{1, 2, 3, 4, 5, 6}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestZerosOnes(t *testing.T) {
	z := Zeros([]int{2, 3})
	for _, v := range z.Data() {
		if v != 0 {
			t.Fatal("zeros contains non-zero")
		}
	}
	o := Ones([]int{2, 3})
	for _, v := range o.Data() {
		if v != 1 {
			t.Fatal("ones contains non-one")
		}
	}
}

func TestLazyEvaluation(t *testing.T) {
	a := FromFloat32([]float32{1, 2}, []int{2})
	b := FromFloat32([]float32{3, 4}, []int{2})
	c := a.Add(b) // lazy — no computation yet
	if c.uop.buf != nil {
		t.Fatal("expected lazy (nil buf)")
	}
	c.Realize() // triggers computation
	if c.uop.buf == nil {
		t.Fatal("expected realized (non-nil buf)")
	}
	got := c.Data()
	if got[0] != 4 || got[1] != 6 {
		t.Fatalf("got %v want [4 6]", got)
	}
}

func BenchmarkAddMul(b *testing.B) {
	n := 1024 * 1024
	x := Rand([]int{n})
	y := Rand([]int{n})
	x.Realize()
	y.Realize()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z := x.Add(y).Mul(x)
		z.Realize()
	}
}
