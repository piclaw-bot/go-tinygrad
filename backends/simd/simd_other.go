//go:build !amd64 && !arm64

package simd

import "unsafe"

const hasSgemmAsm = false

func Sdot(x, y []float32) float32 { return sdotScalar(x, y) }

func Saxpy(alpha float32, x []float32, y []float32) { saxpyScalar(alpha, x, y) }

// SgemmNT is a safe no-op on architectures without SIMD SGEMM assembly.
// Public callers are expected to check HasSgemmAsm, but the fallback must not
// panic because backends/simd is a facade with defensive no-op behavior.
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {}

// SgemmNN is a safe no-op on architectures without SIMD SGEMM assembly.
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {}
