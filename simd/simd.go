package simd

// sdot computes the dot product of two float32 slices using SIMD when available.
// Falls back to scalar code on unsupported architectures.
func Sdot(x, y []float32) float32

// saxpy computes y[i] += alpha * x[i] using SIMD when available.
func Saxpy(alpha float32, x []float32, y []float32)
