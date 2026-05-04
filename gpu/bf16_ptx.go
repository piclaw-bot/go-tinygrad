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
var fnBF16VecAdd CUfunction
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

// BF16 PTX kernels

// BF16RMSNormPTX: RMSNorm on BF16 data with BF16 weights.
// One block, 256 threads. Phase 1: sum-of-squares in F32. Phase 2: scale+narrow.
// BF16RMSNormPTX: emulated BF16 via uint16 load + shift (works on any sm_80+)
var BF16RMSNormPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry bf16_rms_norm(
    .param .u64 x,
    .param .u64 w,
    .param .u32 N,
    .param .f32 eps
) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [N];
    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [w];
    ld.param.f32 %f5, [eps];

    // Phase 1: sum of squares (load u16, shift to F32)
    mov.f32 %f0, 0f00000000;
    mov.u32 %r2, %r0;
ss_loop:
    setp.ge.u32 %p, %r2, %r1;
    @%p bra ss_reduce;
    mul.wide.u32 %rd2, %r2, 2;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.u16 %r4, [%rd3];
    shl.b32 %r4, %r4, 16;
    mov.b32 %f1, %r4;
    fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r2, %r2, 256;
    bra ss_loop;

ss_reduce:
    mov.u64 %rd4, sdata;
    mul.wide.u32 %rd5, %r0, 4;
    add.u64 %rd4, %rd4, %rd5;
    st.shared.f32 [%rd4], %f0;
    bar.sync 0;

    mov.u32 %r3, 128;
red_loop:
    setp.lt.u32 %p, %r3, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r0, %r3;
    @%p bra red_skip;
    add.u32 %r5, %r0, %r3;
    mul.wide.u32 %rd5, %r5, 4;
    mov.u64 %rd6, sdata;
    add.u64 %rd6, %rd6, %rd5;
    ld.shared.f32 %f1, [%rd6];
    ld.shared.f32 %f2, [%rd4];
    add.f32 %f2, %f2, %f1;
    st.shared.f32 [%rd4], %f2;
red_skip:
    bar.sync 0;
    shr.u32 %r3, %r3, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r0, 0;
    @%p bra apply_wait;
    ld.shared.f32 %f0, [sdata];
    cvt.rn.f32.u32 %f1, %r1;
    div.rn.f32 %f0, %f0, %f1;
    add.f32 %f0, %f0, %f5;
    rsqrt.approx.f32 %f0, %f0;
    mul.f32 %f1, %f0, %f0;
    mul.f32 %f1, %f1, %f0;
    neg.f32 %f1, %f1;
    fma.rn.f32 %f0, %f0, 0f40400000, %f1;
    mul.f32 %f0, %f0, 0f3F000000;
    st.shared.f32 [sdata], %f0;

apply_wait:
    bar.sync 0;
    ld.shared.f32 %f3, [sdata];

    mov.u32 %r2, %r0;
apply_loop:
    setp.ge.u32 %p, %r2, %r1;
    @%p bra done;
    mul.wide.u32 %rd2, %r2, 2;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.u16 %r4, [%rd3];
    shl.b32 %r4, %r4, 16;
    mov.b32 %f1, %r4;
    add.u64 %rd6, %rd1, %rd2;
    ld.global.u16 %r5, [%rd6];
    shl.b32 %r5, %r5, 16;
    mov.b32 %f2, %r5;
    mul.f32 %f1, %f1, %f3;
    mul.f32 %f1, %f1, %f2;
    mov.b32 %r4, %f1;
    shr.u32 %r4, %r4, 16;
    st.global.u16 [%rd3], %r4;
    add.u32 %r2, %r2, 256;
    bra apply_loop;

done:
    ret;
}
`

// BF16VecAddPTX: dst[i] = BF16(F32(a[i]) + F32(b[i]))
var BF16VecAddPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry bf16_vec_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 dst,
    .param .u32 N
) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<6>;
    .reg .f32 %f<4>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r1, [N];
    setp.ge.u32 %p, %r3, %r1;
    @%p bra done;

    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [b];
    ld.param.u64 %rd2, [dst];

    mul.wide.u32 %rd3, %r3, 2;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.u16 %r3, [%rd4];
    shl.b32 %r3, %r3, 16;
    mov.b32 %f0, %r3;

    add.u64 %rd4, %rd1, %rd3;
    ld.global.u16 %r3, [%rd4];
    shl.b32 %r3, %r3, 16;
    mov.b32 %f1, %r3;

    add.f32 %f2, %f0, %f1;
    mov.b32 %r3, %f2;
    shr.u32 %r3, %r3, 16;
    add.u64 %rd4, %rd2, %rd3;
    st.global.u16 [%rd4], %r3;

done:
    ret;
}
`
