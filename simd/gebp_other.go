//go:build !amd64 && !arm64

package simd

import "unsafe"

const gebpMR = 4

func gebpMicroKernel(k int, alpha float32, a unsafe.Pointer, lda int, bp unsafe.Pointer, c unsafe.Pointer, ldc int) {
	panic("gebpMicroKernel: no assembly for this architecture")
}
