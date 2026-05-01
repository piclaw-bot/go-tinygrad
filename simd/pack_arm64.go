package simd

// packBNTAsm packs B[jj:jj+16, 0:k] into bp[k, 16] using NEON.
// Falls back to Go scalar for partial tiles (nr < 16).
//
//go:noescape
func packBNTAsm(
	b0, b1, b2, b3, b4, b5, b6, b7,
	b8, b9, b10, b11, b12, b13, b14, b15 uintptr,
	k int, bp uintptr)
