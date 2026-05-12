package simd

// Vector operations for inference hot paths.
// Public entrypoints are implemented by architecture-specific dispatch files
// plus scalar fallback files. This file holds shared scalar helpers.

import "math"

// HasVecAsm is true if vector assembly kernels are available at runtime.
var HasVecAsm bool

// Go fallback implementations (used by vec_other.go or when assembly not available)
func snrm2Go(x []float32) float32 {
	ss := float32(0)
	for _, v := range x {
		ss += v * v
	}
	return float32(math.Sqrt(float64(ss)))
}

func vecAddGo(dst, a, b []float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func vecMulGo(dst, a, b []float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func vecScaleAddGo(dst, a, b []float32, scale float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		dst[i] = a[i] + scale*b[i]
	}
}

func vecScaleGo(dst, a []float32, scale float32) {
	n := len(dst)
	if len(a) < n {
		n = len(a)
	}
	for i := 0; i < n; i++ {
		dst[i] = a[i] * scale
	}
}

func vecSiLUMulGo(dst, a, b []float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		x := a[i]
		s := x / (1.0 + float32(math.Exp(float64(-x))))
		dst[i] = s * b[i]
	}
}

func geluTanhMulGo(dst, a, b []float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		x := a[i]
		x3 := x * x * x
		inner := float32(0.7978845608) * (x + 0.044715*x3)
		tanh := float32(math.Tanh(float64(inner)))
		dst[i] = 0.5 * x * (1.0 + tanh) * b[i]
	}
}

func rmsNormGo(x, w []float32, eps float32) {
	n := len(x)
	if n == 0 || len(w) < n {
		return
	}
	ss := float32(0)
	for _, v := range x {
		ss += v * v
	}
	ss = 1.0 / float32Sqrt(ss/float32(n)+eps)
	for i := range x {
		x[i] = w[i] * x[i] * ss
	}
}

func rmsNormBF16Go(x, w []float32, eps float32) {
	rmsNormGo(x, w, eps)
	toBF16Go(x)
}

func rmsNormNoScaleGo(x []float32, eps float32) {
	n := len(x)
	if n == 0 {
		return
	}
	ss := float32(0)
	for _, v := range x {
		ss += v * v
	}
	ss = float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	for i := range x {
		x[i] *= ss
	}
}

func toBF16Go(x []float32) {
	for i := range x {
		x[i] = toBF16Single(x[i])
	}
}

func toBF16Single(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) & 0xFFFF0000)
}

func float32Sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func bf16WidenToF32Go(dst []float32, src []uint16) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] = BF16ToF32(src[i])
	}
}

func bf16NarrowFromF32Go(dst []uint16, src []float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] = F32ToBF16(src[i])
	}
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}
