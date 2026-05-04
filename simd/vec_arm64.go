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

	BF16VecAdd(dst, a, b)
}

	BF16RMSNorm(x, w, eps)
}
