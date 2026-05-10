//go:build arm64

package simd

//go:noescape
func sdotAsm(x, y []float32) float32

//go:noescape
func saxpyAsm(alpha float32, x []float32, y []float32)

// Sdot computes the dot product of two float32 slices using NEON when
// available, falling back to scalar code otherwise.
func Sdot(x, y []float32) float32 {
	// Assembly kernels assume equal lengths. Preserve scalar fallback semantics
	// for defensive callers that pass mismatched slices.
	if len(x) == len(y) && HasDotAsm {
		return sdotAsm(x, y)
	}
	return sdotScalar(x, y)
}

// Saxpy computes y[i] += alpha*x[i] using NEON when available.
func Saxpy(alpha float32, x []float32, y []float32) {
	if len(x) == len(y) && HasDotAsm {
		saxpyAsm(alpha, x, y)
		return
	}
	saxpyScalar(alpha, x, y)
}
