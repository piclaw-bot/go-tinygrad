package simd

//go:noescape
func packBNTAsm(
	b0, b1, b2, b3, b4, b5, b6, b7,
	b8, b9, b10, b11, b12, b13, b14, b15 uintptr,
	k int, bp uintptr)

const hasAvxPack = false
const hasNeonPack = false
