package simd

import "unsafe"

//go:noescape
func sgemmNTTileFMA(iLen, jLen, kLen int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, ldb int, c unsafe.Pointer, ldc int)
