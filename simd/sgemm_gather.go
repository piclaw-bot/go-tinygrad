package simd

import "unsafe"

// SgemmNTGather computes C += alpha * A * B^T using AVX2 VGATHERDPS.
// No packing, no horizontal reductions. Gathers 8 B values from 8 rows
// per k step, broadcasts each A[i,p], and FMAs into C tile accumulators.
//
// MR=6 rows × NR=8 columns per tile. Index vector [0,ldb,2*ldb,...,7*ldb]
// enables VGATHERDPS to read B[jj+0..7, p] in one instruction.
func SgemmNTGather(m, n, k int, alpha float32, aPtr, bPtr, cPtr unsafe.Pointer, lda, ldb, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	const MR = 6
	const NR = 8

	// Build gather index: [0, ldb, 2*ldb, ..., 7*ldb] (byte offsets / 4 = element offsets)
	indices := [8]int32{
		0, int32(ldb), int32(2 * ldb), int32(3 * ldb),
		int32(4 * ldb), int32(5 * ldb), int32(6 * ldb), int32(7 * ldb),
	}

	a := unsafe.Slice((*float32)(aPtr), m*lda)
	b := (*float32)(bPtr) // raw pointer for gather
	c := unsafe.Slice((*float32)(cPtr), m*ldc)

	for jj := 0; jj < n; jj += NR {
		nr := NR
		if jj+nr > n {
			nr = n - jj
		}
		for ii := 0; ii < m; ii += MR {
			mr := MR
			if ii+mr > m {
				mr = m - ii
			}
			if nr == NR {
				if mr == MR {
					gatherMicroKernel6x8(k, alpha,
						unsafe.Pointer(&a[ii*lda]), lda,
						unsafe.Add(unsafe.Pointer(b), 4*jj*ldb),
						unsafe.Pointer(&indices[0]),
						unsafe.Pointer(&c[ii*ldc+jj]), ldc)
				} else {
					// Edge rows: temp buffer, call full kernel, copy back
					var tmp [MR * NR]float32
					for i := 0; i < mr; i++ {
						copy(tmp[i*NR:i*NR+NR], c[(ii+i)*ldc+jj:(ii+i)*ldc+jj+NR])
					}
					gatherMicroKernel6x8(k, alpha,
						unsafe.Pointer(&a[ii*lda]), lda,
						unsafe.Add(unsafe.Pointer(b), 4*jj*ldb),
						unsafe.Pointer(&indices[0]),
						unsafe.Pointer(&tmp[0]), NR)
					for i := 0; i < mr; i++ {
						copy(c[(ii+i)*ldc+jj:(ii+i)*ldc+jj+NR], tmp[i*NR:i*NR+NR])
					}
				}
			} else {
				// Edge tile: scalar fallback
				for i := 0; i < mr; i++ {
					for d := 0; d < nr; d++ {
						sum := float32(0)
						bRow := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(b), 4*(jj+d)*ldb)), k)
						for p := 0; p < k; p++ {
							sum += a[(ii+i)*lda+p] * bRow[p]
						}
						c[(ii+i)*ldc+jj+d] += alpha * sum
					}
				}
			}
		}
	}
}
