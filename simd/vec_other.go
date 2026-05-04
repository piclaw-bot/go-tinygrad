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
