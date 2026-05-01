package simd

import (
	"unsafe"
)

const gebpNR = 16

var gebpBuf []float32

func ensureGebpBuf(size int) []float32 {
	if cap(gebpBuf) < size {
		gebpBuf = make([]float32, size)
	}
	return gebpBuf[:size]
}

func packBNT(b []float32, ldb, jj, nr, k int, bp []float32) {
	if nr == gebpNR && (hasNeonPack || hasAvxPack) {
		packBNTAsm(
			uintptr(unsafe.Pointer(&b[(jj+0)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+1)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+2)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+3)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+4)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+5)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+6)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+7)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+8)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+9)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+10)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+11)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+12)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+13)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+14)*ldb])),
			uintptr(unsafe.Pointer(&b[(jj+15)*ldb])),
			k, uintptr(unsafe.Pointer(&bp[0])))
		return
	}
	packBNTScalar(b, ldb, jj, nr, k, bp)
}

func packBNTScalar(b []float32, ldb, jj, nr, k int, bp []float32) {
	d := 0
	for ; d+8 <= nr; d += 8 {
		r0 := b[(jj+d)*ldb:]
		r1 := b[(jj+d+1)*ldb:]
		r2 := b[(jj+d+2)*ldb:]
		r3 := b[(jj+d+3)*ldb:]
		r4 := b[(jj+d+4)*ldb:]
		r5 := b[(jj+d+5)*ldb:]
		r6 := b[(jj+d+6)*ldb:]
		r7 := b[(jj+d+7)*ldb:]
		for p := 0; p < k; p++ {
			off := p*gebpNR + d
			bp[off] = r0[p]
			bp[off+1] = r1[p]
			bp[off+2] = r2[p]
			bp[off+3] = r3[p]
			bp[off+4] = r4[p]
			bp[off+5] = r5[p]
			bp[off+6] = r6[p]
			bp[off+7] = r7[p]
		}
	}
	for ; d < nr; d++ {
		row := b[(jj+d)*ldb:]
		for p := 0; p < k; p++ {
			bp[p*gebpNR+d] = row[p]
		}
	}
	if nr < gebpNR {
		for p := 0; p < k; p++ {
			for dd := nr; dd < gebpNR; dd++ {
				bp[p*gebpNR+dd] = 0
			}
		}
	}
}

func SgemmNTGebp(m, n, k int, alpha float32, aPtr, bPtr, cPtr unsafe.Pointer, lda, ldb, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	a := unsafe.Slice((*float32)(aPtr), m*lda)
	b := unsafe.Slice((*float32)(bPtr), n*ldb)
	c := unsafe.Slice((*float32)(cPtr), m*ldc)
	bp := ensureGebpBuf(k * gebpNR)

	for jj := 0; jj < n; jj += gebpNR {
		nr := gebpNR
		if jj+nr > n {
			nr = n - jj
		}
		packBNT(b, ldb, jj, nr, k, bp)

		for ii := 0; ii < m; ii += gebpMR {
			mr := gebpMR
			if ii+mr > m {
				mr = m - ii
			}
			if nr == gebpNR {
				if mr == gebpMR {
					gebpMicroKernel(k, alpha,
						unsafe.Pointer(&a[ii*lda]),
						lda,
						unsafe.Pointer(&bp[0]),
						unsafe.Pointer(&c[ii*ldc+jj]),
						ldc)
				} else {
					var tmp [gebpMR * gebpNR]float32
					for i := 0; i < mr; i++ {
						copy(tmp[i*gebpNR:i*gebpNR+gebpNR], c[(ii+i)*ldc+jj:(ii+i)*ldc+jj+gebpNR])
					}
					gebpMicroKernel(k, alpha,
						unsafe.Pointer(&a[ii*lda]),
						lda,
						unsafe.Pointer(&bp[0]),
						unsafe.Pointer(&tmp[0]),
						gebpNR)
					for i := 0; i < mr; i++ {
						copy(c[(ii+i)*ldc+jj:(ii+i)*ldc+jj+gebpNR], tmp[i*gebpNR:i*gebpNR+gebpNR])
					}
				}
			} else {
				for i := 0; i < mr; i++ {
					for d := 0; d < nr; d++ {
						sum := float32(0)
						for p := 0; p < k; p++ {
							sum += a[(ii+i)*lda+p] * bp[p*gebpNR+d]
						}
						c[(ii+i)*ldc+jj+d] += alpha * sum
					}
				}
			}
		}
	}
}
