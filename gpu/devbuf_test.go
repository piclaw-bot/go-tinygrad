package gpu

import (
	"math"
	"testing"
)

func TestDevBufAdd(t *testing.T) {
	a := NewDevBufFrom([]float32{1, 2, 3, 4})
	b := NewDevBufFrom([]float32{5, 6, 7, 8})
	out := NewDevBuf(4)

	// CPU path
	DevAdd(out, a, b)
	want := []float32{6, 8, 10, 12}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Fatalf("CPU add[%d]=%v want %v", i, v, want[i])
		}
	}

	// GPU path (if available)
	if SgemmReady() {
		initKernels()
		if kernelsLoaded {
			a.ToGPU()
			b.ToGPU()
			DevAdd(out, a, b)
			Sync()
			for i, v := range out.Data() {
				if v != want[i] {
					t.Fatalf("GPU add[%d]=%v want %v", i, v, want[i])
				}
			}
			t.Log("GPU vec_add: OK")
		}
	}
	t.Log("DevAdd OK")
}

func TestDevBufSiLU(t *testing.T) {
	a := NewDevBufFrom([]float32{-2, -1, 0, 1, 2})
	out := NewDevBuf(5)

	DevSiLU(out, a)
	d := out.Data()
	for i, x := range []float32{-2, -1, 0, 1, 2} {
		want := x / (1.0 + float32(math.Exp(float64(-x))))
		if math.Abs(float64(d[i]-want)) > 0.01 {
			t.Fatalf("silu[%d]=%v want %v", i, d[i], want)
		}
	}

	if SgemmReady() {
		initKernels()
		if kernelsLoaded {
			a.ToGPU()
			DevSiLU(out, a)
			Sync()
			d = out.Data()
			for i, x := range []float32{-2, -1, 0, 1, 2} {
				want := x / (1.0 + float32(math.Exp(float64(-x))))
				if math.Abs(float64(d[i]-want)) > 0.05 {
					t.Fatalf("GPU silu[%d]=%v want %v", i, d[i], want)
				}
			}
			t.Log("GPU vec_silu: OK")
		}
	}
	t.Log("DevSiLU OK")
}

func TestDevBufRMSNorm(t *testing.T) {
	x := NewDevBufFrom([]float32{1, 2, 3, 4})
	w := NewDevBufFrom([]float32{1, 1, 1, 1})
	out := NewDevBuf(4)

	DevRMSNorm(out, x, w, 1e-6)
	d := out.Data()
	// RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
	// scale = 1/RMS ≈ 0.3651
	rms := float32(math.Sqrt(float64((1+4+9+16)/4.0 + 1e-6)))
	for i, v := range d {
		want := float32(i+1) / rms
		if math.Abs(float64(v-want)) > 0.001 {
			t.Fatalf("rmsnorm[%d]=%v want %v", i, v, want)
		}
	}

	if SgemmReady() {
		initKernels()
		if kernelsLoaded {
			x.ToGPU()
			w.ToGPU()
			DevRMSNorm(out, x, w, 1e-6)
			Sync()
			d = out.Data()
			for i, v := range d {
				want := float32(i+1) / rms
				if math.Abs(float64(v-want)) > 0.01 {
					t.Fatalf("GPU rmsnorm[%d]=%v want %v", i, v, want)
				}
			}
			t.Log("GPU rms_norm: OK")
		}
	}
	t.Log("DevRMSNorm OK")
}

func TestDevBufGemv(t *testing.T) {
	// W[3,4] * x[4] = out[3]
	W := NewDevBufFrom([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	x := NewDevBufFrom([]float32{1, 1, 1, 1})
	out := NewDevBuf(3)

	DevGemv(out, x, W, 3, 4)
	want := []float32{10, 26, 42}
	for i, v := range out.Data() {
		if math.Abs(float64(v-want[i])) > 0.01 {
			t.Fatalf("gemv[%d]=%v want %v", i, v, want[i])
		}
	}

	if SgemmReady() {
		x.ToGPU()
		W.ToGPU()
		DevGemv(out, x, W, 3, 4)
		Sync()
		for i, v := range out.Data() {
			if math.Abs(float64(v-want[i])) > 0.01 {
				t.Fatalf("GPU gemv[%d]=%v want %v", i, v, want[i])
			}
		}
		t.Log("GPU gemv: OK")
	}
	t.Log("DevGemv OK")
}

func TestDevBufBoundsDoNotPanic(t *testing.T) {
	short := NewDevBufFrom([]float32{2, 4})
	long := NewDevBufFrom([]float32{10, 20, 30, 40})
	out := NewDevBuf(2)

	DevAdd(out, short, long)
	if got := out.Data(); got[0] != 12 || got[1] != 24 {
		t.Fatalf("bounded add = %v", got)
	}

	DevMul(out, short, long)
	if got := out.Data(); got[0] != 20 || got[1] != 80 {
		t.Fatalf("bounded mul = %v", got)
	}

	DevScale(out, long, 0.5)
	if got := out.Data(); got[0] != 5 || got[1] != 10 {
		t.Fatalf("bounded scale = %v", got)
	}

	DevCopy(out, long)
	if got := out.Data(); got[0] != 10 || got[1] != 20 {
		t.Fatalf("bounded copy = %v", got)
	}

	DevToBF16(long, 100)
	DevSoftmax(long, 100)
	DevGELUTanhMul(short, long, 100)
	_ = long.Slice(-1, 2)
	_ = long.Slice(3, 100)
	if neg := NewDevBuf(-10); neg.Len() != 0 {
		t.Fatalf("negative buffer len=%d, want 0", neg.Len())
	}
}

func TestDevBufGemvMalformedDoesNotPanic(t *testing.T) {
	out := NewDevBuf(1)
	x := NewDevBufFrom([]float32{1})
	w := NewDevBufFrom([]float32{1})
	DevGemv(out, x, w, 2, 2)
	DevGemvNN(out, x, w, 2, 2)
	DevLMHead(out, x, w, 2, 2)
	DevGemv(nil, x, w, 1, 1)
	DevGemvNN(out, nil, w, 1, 1)
	DevLMHead(out, x, nil, 1, 1)
}
