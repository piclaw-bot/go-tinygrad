//go:build amd64 || arm64

package simd

//go:noescape
func snrm2Asm(x []float32) float32

//go:noescape
func vecAddAsm(dst, a, b []float32)

//go:noescape
func vecMulAsm(dst, a, b []float32)

//go:noescape
func vecScaleAddAsm(dst, a, b []float32, scale float32)

//go:noescape
func rmsNormAsm(x, w []float32, eps float32)

//go:noescape
func rmsNormBF16Asm(x, w []float32, eps float32)

//go:noescape
func toBF16Asm(x []float32)

//go:noescape
func rmsNormNoScaleAsm(x []float32, eps float32)

//go:noescape
func vecSiLUMulAsm(dst, a, b []float32)

//go:noescape
func geluTanhMulAsm(dst, a, b []float32)

//go:noescape
func bf16DotAsm(x, y []uint16) float32

//go:noescape
func bf16VecAddAsm(dst, a, b []uint16)

//go:noescape
func bf16RMSNormAsm(x, w []uint16, eps float32)

//go:noescape
func bf16WidenToF32Asm(dst []float32, src []uint16)

//go:noescape
func bf16NarrowFromF32Asm(dst []uint16, src []float32)

// Snrm2 returns sqrt(sum(x[i]*x[i])). NOT the RMS — caller divides by sqrt(n).
func Snrm2(x []float32) float32 {
	if HasVecAsm {
		return snrm2Asm(x)
	}
	return snrm2Go(x)
}

// VecAdd computes dst[i] = a[i] + b[i] for i in 0..len(a)-1. dst may alias a.
func VecAdd(dst, a, b []float32) {
	if len(dst) == len(a) && len(b) == len(a) && HasVecAsm {
		vecAddAsm(dst, a, b)
		return
	}
	vecAddGo(dst, a, b)
}

// VecMul computes dst[i] = a[i] * b[i]. dst may alias a.
func VecMul(dst, a, b []float32) {
	if len(dst) == len(a) && len(b) == len(a) && HasVecAsm {
		vecMulAsm(dst, a, b)
		return
	}
	vecMulGo(dst, a, b)
}

// VecScaleAdd computes dst[i] = a[i] + scale*b[i]. Used for residual + scaled output.
func VecScaleAdd(dst, a, b []float32, scale float32) {
	if len(dst) == len(a) && len(b) == len(a) && HasVecAsm {
		vecScaleAddAsm(dst, a, b, scale)
		return
	}
	vecScaleAddGo(dst, a, b, scale)
}

// VecSiLUMul computes dst[i] = silu(a[i]) * b[i] where silu(x) = x/(1+exp(-x)).
// The current assembly stubs call back to the Go math implementation, so dispatch
// directly to Go until an actual polynomial SIMD approximation lands.
func VecSiLUMul(dst, a, b []float32) { vecSiLUMulGo(dst, a, b) }

// GELUTanhMul computes dst[i] = gelu_tanh(a[i]) * b[i].
func GELUTanhMul(dst, a, b []float32) { geluTanhMulGo(dst, a, b) }

// RMSNorm computes x[i] = w[i] * x[i] / rms(x) in-place.
func RMSNorm(x, w []float32, eps float32) {
	if len(x) == len(w) && HasVecAsm {
		rmsNormAsm(x, w, eps)
		return
	}
	rmsNormGo(x, w, eps)
}

// RMSNormBF16 is RMSNorm with each output rounded to BF16 precision.
func RMSNormBF16(x, w []float32, eps float32) {
	if len(x) == len(w) && HasVecAsm {
		rmsNormBF16Asm(x, w, eps)
		return
	}
	rmsNormBF16Go(x, w, eps)
}

// RMSNormNoScale normalizes x in-place by dividing by RMS, without weight.
func RMSNormNoScale(x []float32, eps float32) {
	if HasVecAsm {
		rmsNormNoScaleAsm(x, eps)
		return
	}
	rmsNormNoScaleGo(x, eps)
}

// ToBF16 rounds each element to BF16 precision in-place.
func ToBF16(x []float32) {
	if HasVecAsm {
		toBF16Asm(x)
		return
	}
	toBF16Go(x)
}

// BF16DotAsm computes dot product of two BF16 slices, accumulating in F32.
func BF16DotAsm(x, y []uint16) float32 {
	if len(x) == len(y) && HasVecAsm {
		return bf16DotAsm(x, y)
	}
	return BF16Dot(x, y)
}

// BF16RMSNormAsm computes RMSNorm in-place on BF16 data with BF16 weights.
func BF16RMSNormAsm(x, w []uint16, eps float32) {
	if len(x) == len(w) && HasVecAsm {
		bf16RMSNormAsm(x, w, eps)
		return
	}
	BF16RMSNorm(x, w, eps)
}

// BF16VecAddAsm computes dst = a + b for BF16 slices.
func BF16VecAddAsm(dst, a, b []uint16) {
	if len(dst) == len(a) && len(b) == len(a) && HasVecAsm {
		bf16VecAddAsm(dst, a, b)
		return
	}
	BF16VecAdd(dst, a, b)
}

// BF16WidenToF32 converts []uint16 BF16 to []float32 using SIMD when available.
func BF16WidenToF32(dst []float32, src []uint16) {
	if len(dst) == len(src) && HasVecAsm {
		bf16WidenToF32Asm(dst, src)
		return
	}
	bf16WidenToF32Go(dst, src)
}

// BF16NarrowFromF32 converts []float32 to []uint16 BF16 using SIMD when available.
func BF16NarrowFromF32(dst []uint16, src []float32) {
	if len(dst) == len(src) && HasVecAsm {
		bf16NarrowFromF32Asm(dst, src)
		return
	}
	bf16NarrowFromF32Go(dst, src)
}
