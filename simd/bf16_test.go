package simd

import (
	"math"
	"testing"
)

func TestBF16Convert(t *testing.T) {
	// Round-trip: F32 → BF16 → F32
	vals := []float32{1.0, -0.5, 3.14159, 0.001, 1000.0, 0}
	for _, v := range vals {
		b := F32ToBF16(v)
		back := BF16ToF32(b)
		// BF16 has 7 mantissa bits → ~1% precision
		if v != 0 && math.Abs(float64(back-v)/float64(v)) > 0.01 {
			t.Errorf("BF16 round-trip: %f → %04x → %f (%.2f%% error)", v, b, back, 100*math.Abs(float64(back-v)/float64(v)))
		}
	}
	t.Log("BF16 convert: OK")
}

func TestBF16Dot(t *testing.T) {
	x := BF16FromF32Slice([]float32{1, 2, 3, 4})
	y := BF16FromF32Slice([]float32{0.5, 0.5, 0.5, 0.5})
	got := BF16Dot(x, y)
	// 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 5.0
	if math.Abs(float64(got)-5.0) > 0.1 {
		t.Fatalf("BF16Dot=%f want 5.0", got)
	}
	t.Logf("BF16Dot: %f (want 5.0)", got)
}

func TestBF16DotF32(t *testing.T) {
	x := BF16FromF32Slice([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	y := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	got := BF16DotF32(x, y)
	// Reference in F32
	want := float32(0)
	for i := range y {
		want += BF16ToF32(x[i]) * y[i]
	}
	if math.Abs(float64(got-want)) > 0.01 {
		t.Fatalf("BF16DotF32=%f want %f", got, want)
	}
	t.Logf("BF16DotF32: %f", got)
}

func TestBF16RMSNorm(t *testing.T) {
	x := BF16FromF32Slice([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	w := BF16FromF32Slice([]float32{1, 1, 1, 1, 1, 1, 1, 1})
	eps := float32(1e-6)

	// Reference: RMS then normalize
	xf := BF16ToF32Slice(x)
	ss := float32(0)
	for _, v := range xf {
		ss += v * v
	}
	invRMS := float32(1.0 / math.Sqrt(float64(ss/8.0+eps)))

	BF16RMSNorm(x, w, eps)

	for i := range x {
		got := BF16ToF32(x[i])
		want := xf[i] * invRMS
		if math.Abs(float64(got-want)) > 0.05 {
			t.Fatalf("BF16RMSNorm[%d]=%f want %f", i, got, want)
		}
	}
	t.Log("BF16RMSNorm: OK")
}

func TestBF16VecAdd(t *testing.T) {
	a := BF16FromF32Slice([]float32{1, 2, 3, 4})
	b := BF16FromF32Slice([]float32{10, 20, 30, 40})
	dst := make([]uint16, 4)
	BF16VecAdd(dst, a, b)
	for i := range dst {
		got := BF16ToF32(dst[i])
		want := BF16ToF32(a[i]) + BF16ToF32(b[i])
		if math.Abs(float64(got-want)) > 0.1 {
			t.Fatalf("BF16VecAdd[%d]=%f want %f", i, got, want)
		}
	}
	t.Log("BF16VecAdd: OK")
}

func BenchmarkBF16Dot(b *testing.B) {
	x := BF16FromF32Slice(make([]float32, 3584))
	y := BF16FromF32Slice(make([]float32, 3584))
	for i := range x {
		x[i] = F32ToBF16(float32(i) * 0.001)
		y[i] = F32ToBF16(float32(i) * 0.002)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BF16Dot(x, y)
	}
}

func BenchmarkBF16DotF32(b *testing.B) {
	x := BF16FromF32Slice(make([]float32, 3584))
	y := make([]float32, 3584)
	for i := range x {
		x[i] = F32ToBF16(float32(i) * 0.001)
		y[i] = float32(i) * 0.002
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BF16DotF32(x, y)
	}
}

func BenchmarkBF16RMSNorm(b *testing.B) {
	x := BF16FromF32Slice(make([]float32, 3584))
	w := BF16FromF32Slice(make([]float32, 3584))
	for i := range x {
		x[i] = F32ToBF16(float32(i)*0.001 - 1.0)
		w[i] = F32ToBF16(1.0)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BF16RMSNorm(x, w, 1e-6)
	}
}

func TestBF16DotAsm(t *testing.T) {
	x := BF16FromF32Slice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17})
	y := BF16FromF32Slice([]float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5})
	got := BF16DotAsm(x, y)
	// sum(1..17)*0.5 = 153*0.5 = 76.5
	ref := BF16Dot(x, y)
	if got != ref {
		t.Fatalf("BF16DotAsm=%f ref=%f", got, ref)
	}
	t.Logf("BF16DotAsm: %f (ref=%f, HasVecAsm=%v)", got, ref, HasVecAsm)
}

func TestBF16VecAddAsm(t *testing.T) {
	a := BF16FromF32Slice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	b := BF16FromF32Slice([]float32{10, 20, 30, 40, 50, 60, 70, 80, 90})
	dst := make([]uint16, len(a))
	ref := make([]uint16, len(a))
	BF16VecAddAsm(dst, a, b)
	BF16VecAdd(ref, a, b)
	for i := range dst {
		if dst[i] != ref[i] {
			t.Fatalf("BF16VecAddAsm[%d]=%04x ref=%04x", i, dst[i], ref[i])
		}
	}
	t.Log("BF16VecAddAsm: OK")
}

func TestBF16WidenNarrow(t *testing.T) {
	src := BF16FromF32Slice([]float32{1.0, -0.5, 3.14, 0.001, 1000, 0, -1, 2.5, 99.9})
	f32 := make([]float32, len(src))
	BF16WidenToF32(f32, src)
	for i := range src {
		want := BF16ToF32(src[i])
		if f32[i] != want {
			t.Fatalf("Widen[%d]=%f want %f", i, f32[i], want)
		}
	}
	back := make([]uint16, len(f32))
	BF16NarrowFromF32(back, f32)
	for i := range src {
		if back[i] != src[i] {
			t.Fatalf("Narrow[%d]=%04x want %04x", i, back[i], src[i])
		}
	}
	t.Log("BF16 Widen/Narrow: OK")
}

func BenchmarkBF16DotAsm(b *testing.B) {
	x := BF16FromF32Slice(make([]float32, 3584))
	y := BF16FromF32Slice(make([]float32, 3584))
	for i := range x { x[i] = F32ToBF16(float32(i)*0.001); y[i] = F32ToBF16(float32(i)*0.002) }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BF16DotAsm(x, y)
	}
}

func BenchmarkBF16RMSNormAsm(b *testing.B) {
	x := BF16FromF32Slice(make([]float32, 3584))
	w := BF16FromF32Slice(make([]float32, 3584))
	for i := range x { x[i] = F32ToBF16(float32(i)*0.001-1.0); w[i] = F32ToBF16(1.0) }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BF16RMSNormAsm(x, w, 1e-6)
	}
}

func BenchmarkBF16WidenToF32(b *testing.B) {
	src := BF16FromF32Slice(make([]float32, 3584))
	dst := make([]float32, 3584)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BF16WidenToF32(dst, src)
	}
}
