package ptx

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

// BF16RMSNormNoScalePTX: RMSNormNoScale on BF16 data.
// x[i] = BF16(F32(x[i]) * invRMS), no learned weight.
var BF16RMSNormNoScalePTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry bf16_rms_norm_no_scale(
    .param .u64 x,
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
    ld.param.f32 %f5, [eps];

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
    mul.f32 %f1, %f1, %f3;
    mov.b32 %r4, %f1;
    shr.u32 %r4, %r4, 16;
    st.global.u16 [%rd3], %r4;
    add.u32 %r2, %r2, 256;
    bra apply_loop;

done:
    ret;
}
`

// BF16SiLUMulPTX: dst[i] = BF16(SiLU(F32(gate[i])) * F32(up[i])).
var BF16SiLUMulPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry bf16_silu_mul(
    .param .u64 gate,
    .param .u64 up,
    .param .u64 dst,
    .param .u32 N
) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra done;

    ld.param.u64 %rd0, [gate];
    ld.param.u64 %rd1, [up];
    ld.param.u64 %rd2, [dst];
    mul.wide.u32 %rd3, %r3, 2;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;

    ld.global.u16 %r5, [%rd4];
    shl.b32 %r5, %r5, 16;
    mov.b32 %f0, %r5;
    ld.global.u16 %r6, [%rd5];
    shl.b32 %r6, %r6, 16;
    mov.b32 %f1, %r6;

    // silu(x) = x / (1 + exp(-x)); exp via ex2(-x * log2(e))
    neg.f32 %f2, %f0;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f3, %f2;
    add.f32 %f4, %f3, 0f3F800000;
    div.rn.f32 %f5, %f0, %f4;
    mul.f32 %f6, %f5, %f1;

    mov.b32 %r5, %f6;
    shr.u32 %r5, %r5, 16;
    st.global.u16 [%rd6], %r5;

done:
    ret;
}
`

// BF16GELUTanhMulPTX: gate[i] = BF16(GELUTanh(F32(gate[i])) * F32(up[i])).
var BF16GELUTanhMulPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry bf16_gelu_tanh_mul(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u32 param_n
) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<6>;
    .reg .f32 %f<16>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r0, %r0, %r2, %r1;

    ld.param.u32 %r3, [param_n];
    setp.ge.u32 %p, %r0, %r3;
    @%p bra done;

    ld.param.u64 %rd0, [param_gate];
    ld.param.u64 %rd1, [param_up];
    mul.wide.u32 %rd2, %r0, 2;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;

    ld.global.u16 %r4, [%rd3];
    shl.b32 %r4, %r4, 16;
    mov.b32 %f0, %r4;
    ld.global.u16 %r5, [%rd4];
    shl.b32 %r5, %r5, 16;
    mov.b32 %f1, %r5;

    // gelu_tanh(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    mul.f32 %f2, %f0, %f0;
    mul.f32 %f3, %f2, %f0;
    mul.f32 %f3, %f3, 0f3D372713;
    add.f32 %f3, %f0, %f3;
    mul.f32 %f3, %f3, 0f3F4C422A;

    // tanh(z) = 1 - 2/(1+exp(2z)); exp via ex2(2z*log2(e))
    mul.f32 %f4, %f3, 0f4038AA3B;
    ex2.approx.f32 %f4, %f4;
    add.f32 %f5, %f4, 0f3F800000;
    mov.f32 %f6, 0f40000000;
    div.approx.f32 %f6, %f6, %f5;
    mov.f32 %f7, 0f3F800000;
    sub.f32 %f7, %f7, %f6;

    add.f32 %f7, %f7, 0f3F800000;
    mul.f32 %f7, %f7, 0f3F000000;
    mul.f32 %f7, %f0, %f7;
    mul.f32 %f7, %f7, %f1;

    mov.b32 %r4, %f7;
    shr.u32 %r4, %r4, 16;
    st.global.u16 [%rd3], %r4;

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
