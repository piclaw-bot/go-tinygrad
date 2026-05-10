package simd

import "math"

// vecSiLUMulGo is the Go fallback for SiLU(a)*b.
// Called from assembly stubs on both amd64 and arm64.
func vecSiLUMulGo(dst, a, b []float32) {
	for i := range a {
		x := a[i]
		s := x / (1.0 + float32(math.Exp(float64(-x))))
		dst[i] = s * b[i]
	}
}

func init() {
	HasVecAsm = RuntimeCapabilities().HasVec
}
