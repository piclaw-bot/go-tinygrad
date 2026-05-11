//go:build !amd64 && !arm64

package simd

import "unsafe"

func sgemmNTTileFMA(iLen, jLen, kLen int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, ldb int, c unsafe.Pointer, ldc int) {
	panic("sgemmNTTileFMA: no assembly for this architecture")
}
