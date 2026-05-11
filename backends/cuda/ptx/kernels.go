package ptx

// Individual PTX kernels for LLM inference.
// Split into separate modules for reliability.

const VecAddPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_add(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<4>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [C];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd5];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd6], %f2;
done: ret;
}
`

const VecMulPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_mul(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<4>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [C];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd5];
    mul.f32 %f2, %f0, %f1;
    st.global.f32 [%rd6], %f2;
done: ret;
}
`

const VecSiLUPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_silu(.param .u64 A, .param .u64 B, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<6>; .reg .f32 %f<8>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];
    neg.f32 %f1, %f0;
    mul.f32 %f1, %f1, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f1;
    add.f32 %f3, %f2, 0f3F800000;
    div.rn.f32 %f4, %f0, %f3;
    st.global.f32 [%rd5], %f4;
done: ret;
}
`

const VecScalePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_scale(.param .u64 A, .param .u64 B, .param .f32 S, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<6>; .reg .f32 %f<4>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.f32 %f2, [S];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];
    mul.f32 %f1, %f0, %f2;
    st.global.f32 [%rd5], %f1;
done: ret;
}
`

// Truncate F32 values in-place to BF16 precision by clearing the low 16 mantissa bits.
const ToBF16F32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry to_bf16_f32(.param .u64 A, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<4>; .reg .f32 %f<2>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A];
    mul.wide.u32 %rd1, %r3, 4; add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f0, [%rd2];
    mov.b32 %r5, %f0;
    shr.u32 %r5, %r5, 16;
    shl.b32 %r5, %r5, 16;
    mov.b32 %f1, %r5;
    st.global.f32 [%rd2], %f1;
done: ret;
}
`

const RmsNormPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry rms_norm(.param .u64 A, .param .u64 W, .param .u64 B, .param .u32 N, .param .f32 eps) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<8>; .reg .pred %p;
    .shared .align 4 .f32 sdata[256];
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [N];
    ld.param.u64 %rd0, [A];
    mov.f32 %f0, 0f00000000;
    mov.u32 %r2, %r0;
L1: setp.ge.u32 %p, %r2, %r1; @%p bra L2;
    mul.wide.u32 %rd3, %r2, 4; add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4]; fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r2, %r2, 256; bra L1;
L2: mul.wide.u32 %rd5, %r0, 4; mov.u64 %rd6, sdata; add.u64 %rd5, %rd6, %rd5;
    st.shared.f32 [%rd5], %f0; bar.sync 0;
    mov.u32 %r3, 128;
L3: setp.lt.u32 %p, %r3, 1; @%p bra L4;
    setp.ge.u32 %p, %r0, %r3; @%p bra L3b;
    mul.wide.u32 %rd5, %r0, 4; add.u64 %rd5, %rd6, %rd5; ld.shared.f32 %f1, [%rd5];
    add.u32 %r4, %r0, %r3; mul.wide.u32 %rd7, %r4, 4; add.u64 %rd7, %rd6, %rd7;
    ld.shared.f32 %f2, [%rd7]; add.f32 %f1, %f1, %f2; st.shared.f32 [%rd5], %f1;
L3b: bar.sync 0; shr.u32 %r3, %r3, 1; bra L3;
L4: setp.ne.u32 %p, %r0, 0; @%p bra L5;
    ld.shared.f32 %f0, [sdata]; cvt.rn.f32.u32 %f3, %r1;
    div.rn.f32 %f0, %f0, %f3; ld.param.f32 %f4, [eps];
    add.f32 %f0, %f0, %f4; rsqrt.approx.f32 %f1, %f0; mul.f32 %f2, %f0, %f1; mul.f32 %f2, %f2, %f1; mul.f32 %f2, %f2, 0fBF000000; add.f32 %f2, %f2, 0f3FC00000; mul.f32 %f0, %f1, %f2; st.shared.f32 [sdata], %f0;
L5: bar.sync 0; ld.shared.f32 %f5, [sdata];
    ld.param.u64 %rd1, [W]; ld.param.u64 %rd2, [B];
    mov.u32 %r2, %r0;
L6: setp.ge.u32 %p, %r2, %r1; @%p bra L7;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd7, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4]; ld.global.f32 %f2, [%rd5];
    mul.f32 %f3, %f1, %f5; mul.f32 %f3, %f3, %f2; st.global.f32 [%rd7], %f3;
    add.u32 %r2, %r2, 256; bra L6;
L7: ret;
}
`

const FusedSiLUMulPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry fused_silu_mul(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<8>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [C];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd5];
    // silu(a) = a / (1 + exp(-a))
    neg.f32 %f2, %f0;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f3, %f2;
    add.f32 %f4, %f3, 0f3F800000;
    div.rn.f32 %f5, %f0, %f4;
    // out = silu(a) * b
    mul.f32 %f6, %f5, %f1;
    st.global.f32 [%rd6], %f6;
done: ret;
}
`

const RmsNormNoScalePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry rms_norm_no_scale(.param .u64 A, .param .u64 B, .param .u32 N, .param .f32 eps) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<8>; .reg .pred %p;
    .shared .align 4 .f32 sdata[256];
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [N];
    ld.param.u64 %rd0, [A];
    mov.f32 %f0, 0f00000000;
    mov.u32 %r2, %r0;
L1ns: setp.ge.u32 %p, %r2, %r1; @%p bra L2ns;
    mul.wide.u32 %rd3, %r2, 4; add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4]; fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r2, %r2, 256; bra L1ns;
L2ns: mul.wide.u32 %rd5, %r0, 4; mov.u64 %rd6, sdata; add.u64 %rd5, %rd6, %rd5;
    st.shared.f32 [%rd5], %f0; bar.sync 0;
    mov.u32 %r3, 128;
L3ns: setp.lt.u32 %p, %r3, 1; @%p bra L4ns;
    setp.ge.u32 %p, %r0, %r3; @%p bra L3bns;
    mul.wide.u32 %rd5, %r0, 4; add.u64 %rd5, %rd6, %rd5; ld.shared.f32 %f1, [%rd5];
    add.u32 %r4, %r0, %r3; mul.wide.u32 %rd7, %r4, 4; add.u64 %rd7, %rd6, %rd7;
    ld.shared.f32 %f2, [%rd7]; add.f32 %f1, %f1, %f2; st.shared.f32 [%rd5], %f1;
L3bns: bar.sync 0; shr.u32 %r3, %r3, 1; bra L3ns;
L4ns: setp.ne.u32 %p, %r0, 0; @%p bra L5ns;
    ld.shared.f32 %f0, [sdata]; cvt.rn.f32.u32 %f3, %r1;
    div.rn.f32 %f0, %f0, %f3; ld.param.f32 %f4, [eps];
    add.f32 %f0, %f0, %f4; rsqrt.approx.f32 %f1, %f0; mul.f32 %f2, %f0, %f1; mul.f32 %f2, %f2, %f1; mul.f32 %f2, %f2, 0fBF000000; add.f32 %f2, %f2, 0f3FC00000; mul.f32 %f0, %f1, %f2; st.shared.f32 [sdata], %f0;
L5ns: bar.sync 0; ld.shared.f32 %f5, [sdata];
    ld.param.u64 %rd2, [B];
    mov.u32 %r2, %r0;
L6ns: setp.ge.u32 %p, %r2, %r1; @%p bra L7ns;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd7, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.f32 %f3, %f1, %f5; st.global.f32 [%rd7], %f3;
    add.u32 %r2, %r2, 256; bra L6ns;
L7ns: ret;
}
`

// GELUTanhMulPTX: fused gelu_tanh(gate) * up
// gate[i] = gelu_tanh(gate[i]) * up[i]
// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Approximation: tanh(z) ≈ z * (27 + z^2) / (27 + 9*z^2) for small z; use ex2 for larger
const GELUTanhMulPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry gelu_tanh_mul(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u32 param_n
) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<16>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r0, %r0, %r2, %r1;  // global idx

    ld.param.u32 %r3, [param_n];
    setp.ge.u32 %p, %r0, %r3;
    @%p bra done;

    ld.param.u64 %rd0, [param_gate];
    ld.param.u64 %rd1, [param_up];

    // Load gate[i] and up[i]
    mul.wide.u32 %rd2, %r0, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];      // x = gate[i]
    add.u64 %rd2, %rd1, %rd2;
    ld.global.f32 %f1, [%rd2];      // up[i]

    // gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Let z = sqrt(2/pi) * (x + 0.044715 * x^3)
    // sqrt(2/pi) = 0.7978845608
    mul.f32 %f2, %f0, %f0;          // x^2
    mul.f32 %f3, %f2, %f0;          // x^3
    mul.f32 %f3, %f3, 0f3D372713;   // 0.044715 * x^3
    add.f32 %f3, %f0, %f3;          // x + 0.044715*x^3
    mul.f32 %f3, %f3, 0f3F4C422A;   // z = sqrt(2/pi) * (...)

    // tanh(z) = 1 - 2/(1 + exp(2z))
    // exp(2z) via ex2: exp(2z) = 2^(2z * log2(e))
    mul.f32 %f4, %f3, 0f4038AA3B;   // 2z * log2(e) = 2 * 1.4426950 * z
    ex2.approx.f32 %f4, %f4;         // exp(2z)
    add.f32 %f5, %f4, 0f3F800000;   // 1 + exp(2z)
    mov.f32 %f6, 0f40000000;         // 2.0
    div.approx.f32 %f6, %f6, %f5;   // 2/(1+exp(2z))
    mov.f32 %f7, 0f3F800000;         // 1.0
    sub.f32 %f7, %f7, %f6;          // tanh(z)

    // gelu = 0.5 * x * (1 + tanh)
    add.f32 %f7, %f7, 0f3F800000;   // 1 + tanh(z)
    mul.f32 %f7, %f7, 0f3F000000;   // 0.5 * (1 + tanh)
    mul.f32 %f7, %f0, %f7;          // x * 0.5 * (1 + tanh) = gelu(x)

    // gate[i] = gelu(gate) * up
    mul.f32 %f7, %f7, %f1;
    st.global.f32 [%rd3], %f7;

done:
    ret;
}
`
