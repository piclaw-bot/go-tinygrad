package simd

import "unsafe"

//go:noescape
func gatherMicroKernel6x8(k int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, indices unsafe.Pointer, c unsafe.Pointer, ldc int)
