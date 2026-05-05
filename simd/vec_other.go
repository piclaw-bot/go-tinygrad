//go:build !amd64 && !arm64

package simd

import "math"

func Snrm2(x []float32) float32 {
	ss := float32(0)
	for _, v := range x { ss += v * v }
	return float32(math.Sqrt(float64(ss)))
}

func VecAdd(dst, a, b []float32)  { vecAddGo(dst, a, b) }
func VecMul(dst, a, b []float32)  { vecMulGo(dst, a, b) }

func VecScaleAdd(dst, a, b []float32, scale float32) {
	for i := range a { dst[i] = a[i] + scale*b[i] }
}

func VecSiLUMul(dst, a, b []float32) {
	for i := range a {
		x := a[i]
		s := x / (1.0 + float32(math.Exp(float64(-x))))
		dst[i] = s * b[i]
	}
}

func RMSNorm(x, w []float32, eps float32) { rmsNormGo(x, w, eps) }

func RMSNormNoScale(x []float32, eps float32) {
	n := len(x)
	ss := float32(0)
	for _, v := range x { ss += v * v }
	ss = float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	for i := range x { x[i] *= ss }
}

func GELUTanhMul(dst, a, b []float32) {
	geluTanhMulGo(dst, a, b)
}

func RMSNormBF16(x, w []float32, eps float32) {
	rmsNormGo(x, w, eps)
	ToBF16(x)
}

func ToBF16(x []float32) {
	for i := range x {
		x[i] = toBF16Single(x[i])
	}
}

func toBF16Single(x float32) float32 {
	return float32(math.Float32frombits(math.Float32bits(x) & 0xFFFF0000))
}

func init() { HasVecAsm = false }

func BF16DotAsm(x, y []uint16) float32     { return BF16Dot(x, y) }
func BF16RMSNormAsm(x, w []uint16, eps float32) { BF16RMSNorm(x, w, eps) }
func BF16VecAddAsm(dst, a, b []uint16)      { BF16VecAdd(dst, a, b) }
func BF16WidenToF32(dst []float32, src []uint16) {
	for i, v := range src { dst[i] = BF16ToF32(v) }
}
func BF16NarrowFromF32(dst []uint16, src []float32) {
	for i, v := range src { dst[i] = F32ToBF16(v) }
}
