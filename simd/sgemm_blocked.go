package simd

import "unsafe"

// SgemmNTBlockedFMA computes C += alpha * A * B^T with cache-blocked tiling.
// Go handles the 64×64 block decomposition (keeps B tiles L1-resident).
// Assembly handles the inner (i,j,k) loops per tile with FMA.
func SgemmNTBlockedFMA(m, n, k int, alpha float32, aPtr, bPtr, cPtr unsafe.Pointer, lda, ldb, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	const bs = 64
	a := (*float32)(aPtr)
	b := (*float32)(bPtr)
	c := (*float32)(cPtr)

	for jj := 0; jj < n; jj += bs {
		jLen := bs
		if jj+jLen > n {
			jLen = n - jj
		}
		for kk := 0; kk < k; kk += bs {
			kLen := bs
			if kk+kLen > k {
				kLen = k - kk
			}
			// Tile: C[0:m, jj:jj+jLen] += alpha * A[0:m, kk:kk+kLen] · B[jj:jj+jLen, kk:kk+kLen]^T
			// Both A-tile (m×64 = ~1.8KB) and B-tile (64×64 = 16KB) fit in L1 cache.
			aOff := unsafe.Pointer(unsafe.Add(unsafe.Pointer(a), 4*kk))
			bOff := unsafe.Pointer(unsafe.Add(unsafe.Pointer(b), 4*(jj*ldb+kk)))
			cOff := unsafe.Pointer(unsafe.Add(unsafe.Pointer(c), 4*jj))
			sgemmNTTileFMA(m, jLen, kLen, alpha, aOff, lda, bOff, ldb, cOff, ldc)
		}
	}
}
