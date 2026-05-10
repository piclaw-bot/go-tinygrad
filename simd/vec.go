package simd

// Vector operations for inference hot paths.
// Public entrypoints are implemented by architecture-specific dispatch files
// plus scalar fallback files. This file holds shared scalar helpers.

import (
	"math"
	"unsafe"
)

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
	for i := range a {
		dst[i] = a[i] + b[i]
	}
}

func vecMulGo(dst, a, b []float32) {
	for i := range a {
		dst[i] = a[i] * b[i]
	}
}

func vecScaleAddGo(dst, a, b []float32, scale float32) {
	for i := range a {
		dst[i] = a[i] + scale*b[i]
	}
}

func geluTanhMulGo(dst, a, b []float32) {
	for i := range a {
		x := a[i]
		x3 := x * x * x
		inner := float32(0.7978845608) * (x + 0.044715*x3)
		tanh := float32(math.Tanh(float64(inner)))
		dst[i] = 0.5 * x * (1.0 + tanh) * b[i]
	}
}

func rmsNormGo(x, w []float32, eps float32) {
	n := len(x)
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

// float32Sqrt is a fast sqrt for float32
func float32Sqrt(x float32) float32 {
	return float32(uSqrt(float64(x)))
}

//go:nosplit
func uSqrt(x float64) float64 {
	// Use Go's built-in. The assembly versions use VSQRTSS/FSQRT.
	return unsafeSqrt(x)
}

func unsafeSqrt(x float64) float64 {
	bits := *(*uint64)(unsafe.Pointer(&x))
	if bits == 0 || bits == 0x8000000000000000 {
		return x
	}
	// Newton's method with magic number
	bits = 0x5fe6eb50c7b537a9 - (bits >> 1)
	y := *(*float64)(unsafe.Pointer(&bits))
	y = y * (1.5 - 0.5*x*y*y)
	y = y * (1.5 - 0.5*x*y*y)
	return x * y
}

func bf16WidenToF32Go(dst []float32, src []uint16) {
	for i, v := range src {
		dst[i] = BF16ToF32(v)
	}
}

func bf16NarrowFromF32Go(dst []uint16, src []float32) {
	for i, v := range src {
		dst[i] = F32ToBF16(v)
	}
}
