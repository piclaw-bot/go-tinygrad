//go:build !amd64 && !arm64

package simd

const hasNeonPack = false
const hasAvxPack = false

func packBNTAsm(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 uintptr, k int, bp uintptr) {
	panic("packBNTAsm: not available")
}
