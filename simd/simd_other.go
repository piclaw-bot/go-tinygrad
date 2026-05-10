//go:build !amd64 && !arm64

package simd

import "unsafe"

const hasSgemmAsm = false

func Sdot(x, y []float32) float32 { return sdotScalar(x, y) }

func Saxpy(alpha float32, x []float32, y []float32) { saxpyScalar(alpha, x, y) }

// SgemmNT — scalar fallback, should not be called (caller checks HasSgemmAsm).
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	panic("SgemmNT: no SIMD assembly for this architecture")
}

// SgemmNN — scalar fallback, should not be called.
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	panic("SgemmNN: no SIMD assembly for this architecture")
}
