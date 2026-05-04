package simd

// Vector operations for inference hot paths.
// Each has Go declaration here, assembly in vec_amd64.s / vec_arm64.s,
// and scalar fallback in vec_other.go.

import "unsafe"

// Snrm2 returns sqrt(sum(x[i]*x[i])). NOT the RMS — caller divides by sqrt(n).
func Snrm2(x []float32) float32

// VecAdd computes dst[i] = a[i] + b[i] for i in 0..len(a)-1. dst may alias a.
func VecAdd(dst, a, b []float32)

// VecMul computes dst[i] = a[i] * b[i]. dst may alias a.
func VecMul(dst, a, b []float32)

// VecScaleAdd computes dst[i] = a[i] + scale*b[i]. Used for residual + scaled output.
func VecScaleAdd(dst, a, b []float32, scale float32)

// VecSiLUMul computes dst[i] = silu(a[i]) * b[i] where silu(x) = x/(1+exp(-x)).
func VecSiLUMul(dst, a, b []float32)

// RMSNorm computes x[i] = w[i] * x[i] / rms(x) in-place.
// eps is added inside the sqrt for numerical stability.
func RMSNorm(x, w []float32, eps float32)

// RMSNormBF16 is RMSNorm with each output rounded to BF16 precision.
func RMSNormBF16(x, w []float32, eps float32)

// ToBF16 rounds each element to BF16 precision in-place.
func ToBF16(x []float32)

// HasVecAsm is true if vector assembly kernels are available.
var HasVecAsm bool

// Go fallback implementations (used by vec_other.go or when assembly not available)
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

// BF16 SIMD operations (assembly on amd64/arm64, Go fallback on other)

// BF16DotAsm computes dot product of two BF16 slices, accumulating in F32.
// Uses AVX2: widen BF16→F32 via shift, VFMADD231PS, horizontal reduce.
func BF16DotAsm(x, y []uint16) float32

// BF16RMSNormAsm computes RMSNorm in-place on BF16 data with BF16 weights.
// Phase 1: widen→square→sum in F32. Phase 2: widen→scale→narrow.
func BF16RMSNormAsm(x, w []uint16, eps float32)

// BF16VecAddAsm computes dst = a + b for BF16 slices.
func BF16VecAddAsm(dst, a, b []uint16)

// BF16WidenToF32 converts []uint16 BF16 to []float32 using SIMD.
func BF16WidenToF32(dst []float32, src []uint16)

// BF16NarrowFromF32 converts []float32 to []uint16 BF16 using SIMD.
func BF16NarrowFromF32(dst []uint16, src []float32)
