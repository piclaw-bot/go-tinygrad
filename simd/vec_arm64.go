package simd

import "math"

func vecSiLUMulGo(dst, a, b []float32) {
	for i := range a {
		x := a[i]
		s := x / (1.0 + float32(math.Exp(float64(-x))))
		dst[i] = s * b[i]
	}
}

func init() {
	HasVecAsm = true
}

func bf16VecAddGoFallback(dst, a, b []uint16) {
	BF16VecAdd(dst, a, b)
}

func bf16RMSNormGoFallback(x, w []uint16, eps float32) {
	BF16RMSNorm(x, w, eps)
}
