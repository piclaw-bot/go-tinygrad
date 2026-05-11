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
func DevBF16RMSNorm(x, w *Buffer, n int, eps float32) {
	if fnBF16RMSNorm == 0 {
		return
	}
	EnsureContext()
	nn := uint32(n)
	LaunchKernel(fnBF16RMSNorm, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&w.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&eps))
}

// DevBF16RMSNormNoScale applies RMSNormNoScale on BF16 data: x[i] = BF16(F32(x[i]) * invRMS).
func DevBF16RMSNormNoScale(x *Buffer, n int, eps float32) {
	if fnBF16RMSNormNoScale == 0 {
		return
	}
	EnsureContext()
	nn := uint32(n)
	LaunchKernel(fnBF16RMSNormNoScale, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&eps))
}

// DevBF16VecAdd computes dst[i] = BF16(F32(a[i]) + F32(b[i]))
func DevBF16VecAdd(dst, a, b *Buffer, n int) {
	if fnBF16VecAdd == 0 {
		return
	}
	EnsureContext()
	nn := uint32(n)
	LaunchKernel(fnBF16VecAdd, (nn+255)/256, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&a.Ptr),
		unsafe.Pointer(&b.Ptr),
		unsafe.Pointer(&dst.Ptr),
		unsafe.Pointer(&nn))
}

// DevBF16SiLUMul computes dst[i] = BF16(SiLU(F32(gate[i])) * F32(up[i])).
func DevBF16SiLUMul(dst, gate, up *Buffer, n int) {
	if fnBF16SiLUMul == 0 {
		return
	}
	EnsureContext()
	nn := uint32(n)
	LaunchKernel(fnBF16SiLUMul, (nn+255)/256, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&gate.Ptr),
		unsafe.Pointer(&up.Ptr),
		unsafe.Pointer(&dst.Ptr),
		unsafe.Pointer(&nn))
}

// DevBF16GELUTanhMul computes gate[i] = BF16(GELUTanh(F32(gate[i])) * F32(up[i])) in-place.
func DevBF16GELUTanhMul(gate, up *Buffer, n int) {
	if fnBF16GELUTanhMul == 0 {
		return
	}
	EnsureContext()
	nn := uint32(n)
	LaunchKernel(fnBF16GELUTanhMul, (nn+255)/256, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&gate.Ptr),
		unsafe.Pointer(&up.Ptr),
		unsafe.Pointer(&nn))
}
