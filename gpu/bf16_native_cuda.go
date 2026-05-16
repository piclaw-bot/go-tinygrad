package gpu

// Native BF16 PTX kernels for sm_86+ (Ampere and newer).
//
// Uses native BF16 instructions:
//   ld.global.b16 %h, [addr]  — load BF16 into .b16 register
//   st.global.b16 [addr], %h  — store BF16
//   cvt.f32.bf16 %f, %h       — BF16 → F32 (hardware conversion)
//   cvt.rn.bf16.f32 %h, %f    — F32 → BF16 (hardware rounding)
//
// Benefits over emulated (u16+shift):
//   - 1 instruction per conversion vs 2 (no shl/shr needed)
//   - Hardware rounding (cvt.rn) vs truncation
//   - Potential for BF16 tensor core operations (future)

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/cuda/ptx"
)

var (
	fnNativeBF16RMSNorm CUfunction
	fnNativeBF16VecAdd  CUfunction
	fnNativeBF16Gemv    CUfunction
	nativeBF16Mod       CUmodule
	nativeBF16Ready     bool
)

// InitNativeBF16 loads native BF16 kernels. Call after mega module init.
func InitNativeBF16() {
	if !sgemmReady {
		return
	}
	EnsureContext()

	body := stripPTXHeader(ptx.NativeBF16RMSNormPTX) + "\n" + stripPTXHeader(ptx.NativeBF16VecAddPTX) + "\n" + stripPTXHeader(ptx.NativeBF16GemvPTX)
	full := ".version 7.8\n.target sm_86\n.address_size 64\n\n" + body
	fullBytes := append([]byte(full), 0)

	var mod CUmodule
	r := cuModuleLoadData(&mod, unsafe.Pointer(&fullBytes[0]))
	if r != CUDA_SUCCESS {
		return
	}
	nativeBF16Mod = mod

	extract := func(name string) CUfunction {
		nameBytes := append([]byte(name), 0)
		var fn CUfunction
		cuModuleGetFunction(&fn, mod, unsafe.Pointer(&nameBytes[0]))
		return fn
	}

	fnNativeBF16RMSNorm = extract("native_bf16_rms_norm")
	fnNativeBF16VecAdd = extract("native_bf16_vec_add")
	fnNativeBF16Gemv = extract("native_bf16_gemv")

	if fnNativeBF16RMSNorm != 0 {
		nativeBF16Ready = true
		debugln("[gpu] Native BF16 kernels loaded (PTX 7.8/sm_86)")
	}
}

func NativeBF16Ready() bool { return nativeBF16Ready }

func shutdownNativeBF16() {
	if nativeBF16Mod != 0 && cuModuleUnload != nil {
		EnsureContext()
		cuModuleUnload(nativeBF16Mod)
	}
	nativeBF16Mod = 0
	fnNativeBF16RMSNorm = 0
	fnNativeBF16VecAdd = 0
	fnNativeBF16Gemv = 0
	nativeBF16Ready = false
}

// DevNativeBF16RMSNorm runs hardware BF16 RMSNorm on Ampere+.
func DevNativeBF16RMSNorm(x, w *Buffer, n int, eps float32) bool {
	if !validBF16Buffer(x, n) || !validBF16Buffer(w, n) {
		return false
	}
	if !nativeBF16Ready {
		return DevBF16RMSNorm(x, w, n, eps) // fall back to emulated
	}
	EnsureContext()
	nn := uint32(n)
	if err := LaunchKernel(fnNativeBF16RMSNorm, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr), unsafe.Pointer(&w.Ptr),
		unsafe.Pointer(&nn), unsafe.Pointer(&eps)); err != nil {
		return DevBF16RMSNorm(x, w, n, eps)
	}
	return true
}

// DevNativeBF16VecAdd runs hardware BF16 add on Ampere+.
func DevNativeBF16VecAdd(dst, a, b *Buffer, n int) bool {
	if !validBF16Buffer(dst, n) || !validBF16Buffer(a, n) || !validBF16Buffer(b, n) {
		return false
	}
	if !nativeBF16Ready {
		return DevBF16VecAdd(dst, a, b, n)
	}
	EnsureContext()
	nn := uint32(n)
	if err := LaunchKernel(fnNativeBF16VecAdd, (nn+255)/256, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&a.Ptr), unsafe.Pointer(&b.Ptr),
		unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn)); err != nil {
		return DevBF16VecAdd(dst, a, b, n)
	}
	return true
}
