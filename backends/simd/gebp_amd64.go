package simd

import "unsafe"

const gebpMR = 6

//go:noescape
func gebpMicroKernel(k int, alpha float32, a unsafe.Pointer, lda int, bp unsafe.Pointer, c unsafe.Pointer, ldc int)
