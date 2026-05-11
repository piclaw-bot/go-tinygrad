//go:build !amd64

package simd

import "unsafe"

func gatherMicroKernel6x8(k int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, indices unsafe.Pointer, c unsafe.Pointer, ldc int) {
	panic("gatherMicroKernel6x8: not amd64")
}
