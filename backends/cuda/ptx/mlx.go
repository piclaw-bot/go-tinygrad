package ptx

// MLXGemvPTX is the optimized MLX GEMV kernel.
var MLXGemvPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_gemv(
    .param .u64 x,
    .param .u64 qweight,
    .param .u64 scales,
    .param .u64 biases,
    .param .u64 output,
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize
) {
    .reg .u32 %r<20>;
    .reg .u64 %rd<18>;
    .reg .f32 %f<16>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [outDim];
    ld.param.u32 %r3, [inDim];
    ld.param.u32 %r4, [numGroups];
    ld.param.u32 %r5, [groupSize];

    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;

    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [qweight];
    ld.param.u64 %rd2, [scales];
    ld.param.u64 %rd3, [biases];
    ld.param.u64 %rd4, [output];

    // Weight row pointer: qweight + row * (inDim/8) * 4
    shr.u32 %r6, %r3, 3;
    mul.lo.u32 %r7, %r0, %r6;
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd1, %rd1, %rd5;

    // Scale/bias row pointer
    mul.lo.u32 %r8, %r0, %r4;
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd2, %rd2, %rd6;
    add.u64 %rd3, %rd3, %rd6;

    // Each thread processes packed words tid, tid+256, tid+512...
    // 8 elements per word, coalesced reads across warp
    mov.f32 %f0, 0f00000000;
    mov.u32 %r9, %r1;

pack_loop:
    setp.ge.u32 %p, %r9, %r6;
    @%p bra reduce;

    // Load packed word (coalesced)
    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r10, [%rd8];

    // Base element = word_idx * 8, group = base / groupSize
    shl.b32 %r11, %r9, 3;
    div.u32 %r12, %r11, %r5;
    mul.wide.u32 %rd9, %r12, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f1, [%rd10];
    add.u64 %rd11, %rd3, %rd9;
    ld.global.f32 %f2, [%rd11];

    // x base pointer for 8 elements
    mul.wide.u32 %rd12, %r11, 4;
    add.u64 %rd13, %rd0, %rd12;

    // Unrolled 8 elements: extract 4-bit, dequant, FMA
    // Element 0
    and.b32 %r13, %r10, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 1
    shr.u32 %r13, %r10, 4;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+4];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 2
    shr.u32 %r13, %r10, 8;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+8];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 3
    shr.u32 %r13, %r10, 12;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+12];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 4
    shr.u32 %r13, %r10, 16;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+16];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 5
    shr.u32 %r13, %r10, 20;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+20];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 6
    shr.u32 %r13, %r10, 24;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+24];
    fma.rn.f32 %f0, %f3, %f4, %f0;
    // Element 7
    shr.u32 %r13, %r10, 28;
    cvt.rn.f32.u32 %f3, %r13;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd13+28];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    add.u32 %r9, %r9, 256;
    bra pack_loop;

reduce:
    mov.u64 %rd14, sdata;
    mul.wide.u32 %rd15, %r1, 4;
    add.u64 %rd14, %rd14, %rd15;
    st.shared.f32 [%rd14], %f0;
    bar.sync 0;

    mov.u32 %r14, 128;
red_loop:
    setp.lt.u32 %p, %r14, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r1, %r14;
    @%p bra red_skip;
    add.u32 %r15, %r1, %r14;
    mul.wide.u32 %rd16, %r15, 4;
    mov.u64 %rd17, sdata;
    add.u64 %rd17, %rd17, %rd16;
    ld.shared.f32 %f5, [%rd17];
    ld.shared.f32 %f6, [%rd14];
    add.f32 %f6, %f6, %f5;
    st.shared.f32 [%rd14], %f6;
red_skip:
    bar.sync 0;
    shr.u32 %r14, %r14, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.f32 %f7, [sdata];
    mul.wide.u32 %rd7, %r0, 4;
    add.u64 %rd8, %rd4, %rd7;
    st.global.f32 [%rd8], %f7;

done:
    ret;
}
`

// MLXGemmPTX is the batched MLX GEMM kernel.
var MLXGemmPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_gemm(
    .param .u64 input,
    .param .u64 qweight,
    .param .u64 scales,
    .param .u64 biases,
    .param .u64 output,
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize,
    .param .u32 B
) {
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<12>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r20, %ctaid.y;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [outDim];
    ld.param.u32 %r3, [inDim];
    ld.param.u32 %r4, [numGroups];
    ld.param.u32 %r5, [groupSize];
    ld.param.u32 %r21, [B];

    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;
    setp.ge.u32 %p, %r20, %r21;
    @%p bra done;

    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [qweight];
    ld.param.u64 %rd2, [scales];
    ld.param.u64 %rd3, [biases];
    ld.param.u64 %rd4, [output];

    mul.lo.u32 %r22, %r20, %r3;
    mul.wide.u32 %rd5, %r22, 4;
    add.u64 %rd0, %rd0, %rd5;

    shr.u32 %r7, %r3, 3;
    mul.lo.u32 %r8, %r0, %r7;
    mul.wide.u32 %rd5, %r8, 4;
    add.u64 %rd1, %rd1, %rd5;

    mul.lo.u32 %r9, %r0, %r4;
    mul.wide.u32 %rd6, %r9, 4;
    add.u64 %rd2, %rd2, %rd6;
    add.u64 %rd3, %rd3, %rd6;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r10, %r1;

loop:
    setp.ge.u32 %p, %r10, %r3;
    @%p bra reduce;

    shr.u32 %r11, %r10, 3;
    mul.wide.u32 %rd7, %r11, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r12, [%rd8];

    and.b32 %r13, %r10, 7;
    shl.b32 %r13, %r13, 2;
    shr.u32 %r14, %r12, %r13;
    and.b32 %r14, %r14, 15;
    cvt.rn.f32.u32 %f2, %r14;

    div.u32 %r15, %r10, %r5;
    mul.wide.u32 %rd9, %r15, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f3, [%rd10];
    add.u64 %rd11, %rd3, %rd9;
    ld.global.f32 %f4, [%rd11];

    fma.rn.f32 %f2, %f2, %f3, %f4;

    mul.wide.u32 %rd12, %r10, 4;
    add.u64 %rd13, %rd0, %rd12;
    ld.global.f32 %f5, [%rd13];
    fma.rn.f32 %f1, %f2, %f5, %f1;

    add.u32 %r10, %r10, 256;
    bra loop;

reduce:
    mov.u64 %rd14, sdata;
    mul.wide.u32 %rd15, %r1, 4;
    add.u64 %rd14, %rd14, %rd15;
    st.shared.f32 [%rd14], %f1;
    bar.sync 0;

    mov.u32 %r15, 128;
red_loop:
    setp.lt.u32 %p, %r15, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r1, %r15;
    @%p bra red_skip;
    add.u32 %r16, %r1, %r15;
    mul.wide.u32 %rd7, %r16, 4;
    mov.u64 %rd8, sdata;
    add.u64 %rd8, %rd8, %rd7;
    ld.shared.f32 %f6, [%rd8];
    ld.shared.f32 %f7, [%rd14];
    add.f32 %f7, %f7, %f6;
    st.shared.f32 [%rd14], %f7;
red_skip:
    bar.sync 0;
    shr.u32 %r15, %r15, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.f32 %f8, [sdata];
    mad.lo.u32 %r17, %r20, %r2, %r0;
    mul.wide.u32 %rd7, %r17, 4;
    add.u64 %rd8, %rd4, %rd7;
    st.global.f32 [%rd8], %f8;

done:
    ret;
}
`

// MLXCorrectPTX: bias correction after GPTQ GEMV for MLX weights.
// out[row] += sum_g( sum(x[g*gs:(g+1)*gs]) * correction[row*numGroups+g] )
var MLXCorrectPTX = `
.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_correct(
    .param .u64 x,
    .param .u64 correction,
    .param .u64 output,
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize
) {
    .reg .u32 %r<16>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<8>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [outDim];
    ld.param.u32 %r4, [numGroups];
    ld.param.u32 %r5, [groupSize];

    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;

    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [correction];
    ld.param.u64 %rd2, [output];

    // correction row: correction + row * numGroups
    mul.lo.u32 %r6, %r0, %r4;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd1, %rd1, %rd3;

    // Each thread handles groups tid, tid+256, ...
    mov.f32 %f0, 0f00000000;
    mov.u32 %r7, %r1;

group_loop:
    setp.ge.u32 %p, %r7, %r4;
    @%p bra reduce;

    // Load correction[g]
    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];

    // Sum x in this group: x[g*gs .. (g+1)*gs-1]
    mul.lo.u32 %r8, %r7, %r5;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r9, 0;
xsum:
    setp.ge.u32 %p, %r9, %r5;
    @%p bra xsum_done;
    add.u32 %r10, %r8, %r9;
    mul.wide.u32 %rd6, %r10, 4;
    add.u64 %rd7, %rd0, %rd6;
    ld.global.f32 %f3, [%rd7];
    add.f32 %f2, %f2, %f3;
    add.u32 %r9, %r9, 1;
    bra xsum;
xsum_done:
    fma.rn.f32 %f0, %f2, %f1, %f0;
    add.u32 %r7, %r7, 256;
    bra group_loop;

reduce:
    mov.u64 %rd8, sdata;
    mul.wide.u32 %rd9, %r1, 4;
    add.u64 %rd8, %rd8, %rd9;
    st.shared.f32 [%rd8], %f0;
    bar.sync 0;

    mov.u32 %r11, 128;
red_loop:
    setp.lt.u32 %p, %r11, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r1, %r11;
    @%p bra red_skip;
    add.u32 %r12, %r1, %r11;
    mul.wide.u32 %rd10, %r12, 4;
    mov.u64 %rd11, sdata;
    add.u64 %rd11, %rd11, %rd10;
    ld.shared.f32 %f4, [%rd11];
    ld.shared.f32 %f5, [%rd8];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd8], %f5;
red_skip:
    bar.sync 0;
    shr.u32 %r11, %r11, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.f32 %f6, [sdata];
    mul.wide.u32 %rd4, %r0, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.f32 %f7, [%rd5];
    add.f32 %f7, %f7, %f6;
    st.global.f32 [%rd5], %f7;
done:
    ret;
}
`

// GemvMLXDirect uses the native MLX kernel (no GPTQ transpose).
// Matches CPU GemvMLQ precision. Use for precision-sensitive models.
