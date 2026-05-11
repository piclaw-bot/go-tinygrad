package ptx

// Native BF16 RMSNorm — uses cvt.f32.bf16 / cvt.rn.bf16.f32
var NativeBF16RMSNormPTX = `.version 7.8
.target sm_86
.address_size 64

.visible .entry native_bf16_rms_norm(
    .param .u64 x,
    .param .u64 w,
    .param .u32 N,
    .param .f32 eps
) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .b16 %h<4>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [N];
    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [w];
    ld.param.f32 %f5, [eps];

    // Phase 1: sum of squares with native BF16 load + convert
    mov.f32 %f0, 0f00000000;
    mov.u32 %r2, %r0;
ss_loop:
    setp.ge.u32 %p, %r2, %r1;
    @%p bra ss_reduce;
    mul.wide.u32 %rd2, %r2, 2;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.b16 %h0, [%rd3];
    cvt.f32.bf16 %f1, %h0;
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
    // Newton refinement
    mul.f32 %f1, %f0, %f0;
    mul.f32 %f1, %f1, %f0;
    neg.f32 %f1, %f1;
    fma.rn.f32 %f0, %f0, 0f40400000, %f1;
    mul.f32 %f0, %f0, 0f3F000000;
    st.shared.f32 [sdata], %f0;

apply_wait:
    bar.sync 0;
    ld.shared.f32 %f3, [sdata]; // invRMS

    // Phase 2: native BF16 load → F32 compute → native BF16 store
    mov.u32 %r2, %r0;
apply_loop:
    setp.ge.u32 %p, %r2, %r1;
    @%p bra done;
    mul.wide.u32 %rd2, %r2, 2;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.b16 %h0, [%rd3];
    cvt.f32.bf16 %f1, %h0;
    add.u64 %rd6, %rd1, %rd2;
    ld.global.b16 %h1, [%rd6];
    cvt.f32.bf16 %f2, %h1;
    mul.f32 %f1, %f1, %f3;
    mul.f32 %f1, %f1, %f2;
    cvt.rn.bf16.f32 %h2, %f1;
    st.global.b16 [%rd3], %h2;
    add.u32 %r2, %r2, 256;
    bra apply_loop;

done:
    ret;
}
`

// Native BF16 VecAdd
var NativeBF16VecAddPTX = `.version 7.8
.target sm_86
.address_size 64

.visible .entry native_bf16_vec_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 dst,
    .param .u32 N
) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<6>;
    .reg .f32 %f<4>;
    .reg .b16 %h<4>;
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
    ld.global.b16 %h0, [%rd4];
    cvt.f32.bf16 %f0, %h0;

    add.u64 %rd4, %rd1, %rd3;
    ld.global.b16 %h1, [%rd4];
    cvt.f32.bf16 %f1, %h1;

    add.f32 %f2, %f0, %f1;
    cvt.rn.bf16.f32 %h2, %f2;

    add.u64 %rd4, %rd2, %rd3;
    st.global.b16 [%rd4], %h2;

done:
    ret;
}
`

// Native BF16 GEMV: BF16 input × F32 weights → BF16 output
var NativeBF16GemvPTX = `.version 7.8
.target sm_86
.address_size 64

.visible .entry native_bf16_gemv(
    .param .u64 x,         // [inDim] BF16
    .param .u64 w,         // [outDim, inDim] F32
    .param .u64 output,    // [outDim] BF16
    .param .u32 inDim,
    .param .u32 outDim
) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<8>;
    .reg .b16 %h<2>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;   // row
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [outDim];
    ld.param.u32 %r3, [inDim];

    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;

    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [w];
    ld.param.u64 %rd2, [output];

    // W row offset: w + row * inDim * 4
    mul.lo.u32 %r4, %r0, %r3;
    mul.wide.u32 %rd3, %r4, 4;
    add.u64 %rd1, %rd1, %rd3;

    // Dot product with BF16 x and F32 w
    mov.f32 %f0, 0f00000000;
    mov.u32 %r5, %r1;

dot_loop:
    setp.ge.u32 %p, %r5, %r3;
    @%p bra reduce;

    // Load BF16 x[i] → F32
    mul.wide.u32 %rd4, %r5, 2;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.b16 %h0, [%rd5];
    cvt.f32.bf16 %f1, %h0;

    // Load F32 w[row, i]
    mul.wide.u32 %rd6, %r5, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.f32 %f2, [%rd7];

    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u32 %r5, %r5, 256;
    bra dot_loop;

reduce:
    mov.u64 %rd8, sdata;
    mul.wide.u32 %rd9, %r1, 4;
    add.u64 %rd8, %rd8, %rd9;
    st.shared.f32 [%rd8], %f0;
    bar.sync 0;

    mov.u32 %r6, 128;
red_loop:
    setp.lt.u32 %p, %r6, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r1, %r6;
    @%p bra red_skip;
    add.u32 %r7, %r1, %r6;
    mul.wide.u32 %rd10, %r7, 4;
    mov.u64 %rd11, sdata;
    add.u64 %rd11, %rd11, %rd10;
    ld.shared.f32 %f3, [%rd11];
    ld.shared.f32 %f4, [%rd8];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd8], %f4;
red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;

    ld.shared.f32 %f5, [sdata];
    cvt.rn.bf16.f32 %h1, %f5;
    mul.wide.u32 %rd4, %r0, 2;
    add.u64 %rd5, %rd2, %rd4;
    st.global.b16 [%rd5], %h1;

done:
    ret;
}
`
