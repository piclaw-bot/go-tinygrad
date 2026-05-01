package simd

import "unsafe"

// HasSgemmAsm reports whether SIMD-accelerated SGEMM kernels are available
// on this architecture (amd64 AVX2+FMA, arm64 NEON).
const HasSgemmAsm = hasSgemmAsm

// SgemmNT computes C += alpha * A * B^T.
// A is [m,lda] row-major, B is [n,ldb] row-major, C is [m,ldc] row-major.
// Caller must handle beta (pre-scale C before calling).
//
//go:noescape
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)

// SgemmNN computes C += alpha * A * B.
// A is [m,lda] row-major, B is [k,ldb] row-major, C is [m,ldc] row-major.
// Caller must handle beta.
//
//go:noescape
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)
