package gpu

// BF16 CUDA PTX kernels for sm_80+ (Ampere and newer).
//
// On Ampere+, BF16 is natively supported:
//   - ld.global.b16 / st.global.b16 for BF16 load/store
//   - cvt.f32.bf16 / cvt.bf16.f32 for conversion
//   - FMA in F32 with BF16 inputs
//
// These kernels accept BF16 (uint16) buffers directly,
// halving memory bandwidth vs F32 kernels.

import "unsafe"

var fnBF16RMSNorm CUfunction
var fnBF16RMSNormNoScale CUfunction
var fnBF16VecAdd CUfunction
var fnBF16SiLUMul CUfunction
var fnBF16GELUTanhMul CUfunction
var fnBF16Gemv CUfunction

// DevBF16RMSNorm applies RMSNorm on BF16 data: x[i] = BF16(F32(x[i]) * invRMS * F32(w[i]))
func DevBF16RMSNorm(x, w *Buffer, n int, eps float32) bool {
	if fnBF16RMSNorm == 0 || !validBF16Buffer(x, n) || !validBF16Buffer(w, n) {
		return false
	}
	EnsureContext()
	nn := uint32(n)
	return LaunchKernel(fnBF16RMSNorm, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&w.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&eps)) == nil
}

// DevBF16RMSNormNoScale applies RMSNormNoScale on BF16 data: x[i] = BF16(F32(x[i]) * invRMS).
func DevBF16RMSNormNoScale(x *Buffer, n int, eps float32) bool {
	if fnBF16RMSNormNoScale == 0 || !validBF16Buffer(x, n) {
		return false
	}
	EnsureContext()
	nn := uint32(n)
	return LaunchKernel(fnBF16RMSNormNoScale, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&eps)) == nil
}

// DevBF16VecAdd computes dst[i] = BF16(F32(a[i]) + F32(b[i]))
func DevBF16VecAdd(dst, a, b *Buffer, n int) bool {
	if fnBF16VecAdd == 0 || !validBF16Buffer(dst, n) || !validBF16Buffer(a, n) || !validBF16Buffer(b, n) {
		return false
	}
	EnsureContext()
	grid, okGrid := grid1DFor(n, 256)
	if !okGrid {
		return false
	}
	nn := uint32(n)
	return LaunchKernel(fnBF16VecAdd, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&a.Ptr),
		unsafe.Pointer(&b.Ptr),
		unsafe.Pointer(&dst.Ptr),
		unsafe.Pointer(&nn)) == nil
}

// DevBF16SiLUMul computes dst[i] = BF16(SiLU(F32(gate[i])) * F32(up[i])).
func DevBF16SiLUMul(dst, gate, up *Buffer, n int) bool {
	if fnBF16SiLUMul == 0 || !validBF16Buffer(dst, n) || !validBF16Buffer(gate, n) || !validBF16Buffer(up, n) {
		return false
	}
	EnsureContext()
	grid, okGrid := grid1DFor(n, 256)
	if !okGrid {
		return false
	}
	nn := uint32(n)
	return LaunchKernel(fnBF16SiLUMul, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&gate.Ptr),
		unsafe.Pointer(&up.Ptr),
		unsafe.Pointer(&dst.Ptr),
		unsafe.Pointer(&nn)) == nil
}

// DevBF16GELUTanhMul computes gate[i] = BF16(GELUTanh(F32(gate[i])) * F32(up[i])) in-place.
func DevBF16GELUTanhMul(gate, up *Buffer, n int) bool {
	if fnBF16GELUTanhMul == 0 || !validBF16Buffer(gate, n) || !validBF16Buffer(up, n) {
		return false
	}
	EnsureContext()
	grid, okGrid := grid1DFor(n, 256)
	if !okGrid {
		return false
	}
	nn := uint32(n)
	return LaunchKernel(fnBF16GELUTanhMul, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&gate.Ptr),
		unsafe.Pointer(&up.Ptr),
		unsafe.Pointer(&nn)) == nil
}

func validBF16Buffer(b *Buffer, n int) bool {
	if b == nil || b.Ptr == 0 || n <= 0 || !fitsUint32(n) {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	if n > maxInt/2 {
		return false
	}
	return b.Size >= n*2
}
