package gpu

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
    div.approx.f32 %f4, %f0, %f3;
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
    div.approx.f32 %f0, %f0, %f3; ld.param.f32 %f4, [eps];
    add.f32 %f0, %f0, %f4; rsqrt.approx.f32 %f0, %f0; st.shared.f32 [sdata], %f0;
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
