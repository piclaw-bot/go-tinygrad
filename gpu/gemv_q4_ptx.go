package gpu

// Optimized fused INT4 dequant+GEMV PTX kernel.
//
// Key optimizations (matching tinygrad patterns):
// 1. x[] loaded into shared memory once, reused by all threads in block
// 2. Inner loop fully unrolled over 8 bits per packed int32
// 3. Scale cached per group (group_size=128 means same scale for 128 consecutive inputs)
// 4. Coalesced qweight access (consecutive threads read consecutive columns)
//
// Layout: qweight[inDim/8, outDim], scales[numGroups, outDim], gIdx[inDim]
// Each thread computes one output: out[j] = Σ x[i] * scale[g[i],j] * (qw[i,j]-8)

const GemvQ4OptPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q4sym(
    .param .u64 param_x,
    .param .u64 param_qw,
    .param .u64 param_scales,
    .param .u64 param_gidx,
    .param .u64 param_out,
    .param .u32 param_inDim,
    .param .u32 param_outDim,
    .param .u32 param_numGroups
) {
    .reg .u32 %r<32>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<20>;
    .reg .pred %p<4>;

    // Shared memory for x vector (up to 4096 floats = 16KB)
    .shared .align 16 .f32 sx[4096];

    // j = global output index
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_outDim];
    ld.param.u32 %r5, [param_inDim];
    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_qw];
    ld.param.u64 %rd2, [param_scales];
    ld.param.u64 %rd3, [param_gidx];

    // Phase 1: cooperatively load x[] into shared memory
    // Each thread loads multiple elements: x[tid], x[tid+blockDim], ...
    mov.u32 %r6, %r2;  // tid
load_x:
    setp.ge.u32 %p0, %r6, %r5;
    @%p0 bra load_x_done;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];
    // Store to shared memory
    mov.u64 %rd6, sx;
    add.u64 %rd7, %rd6, %rd4;
    st.shared.f32 [%rd7], %f0;
    add.u32 %r6, %r6, %r1;  // += blockDim
    bra load_x;
load_x_done:
    bar.sync 0;

    // Early exit if j >= outDim
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    // Phase 2: compute dot product
    mov.f32 %f10, 0f00000000;  // accumulator
    shr.u32 %r7, %r5, 3;      // nPacks = inDim / 8
    mov.u32 %r8, 0;            // packIdx
    mov.u64 %rd6, sx;          // shared mem base

    // Precompute base offset for this column: packIdx*outDim + j
    // qweight is [nPacks, outDim], access pattern: qw[packIdx][j]

pack_loop:
    setp.ge.u32 %p0, %r8, %r7;
    @%p0 bra pack_done;

    // Load packed qweight: qw[packIdx * outDim + j]
    mad.lo.u32 %r9, %r8, %r4, %r3;
    mul.wide.u32 %rd8, %r9, 4;
    add.u64 %rd8, %rd1, %rd8;
    ld.global.u32 %r10, [%rd8];  // packed 8x4-bit

    // Base input index: i_base = packIdx * 8
    shl.b32 %r11, %r8, 3;

    // Load gIdx for this pack's first element to get the group
    // (group_size=128, so 8 consecutive elements share the same group most of the time)
    mul.wide.u32 %rd9, %r11, 4;
    add.u64 %rd9, %rd3, %rd9;
    ld.global.u32 %r12, [%rd9];  // group for i_base

    // Load scale for this group and column: scales[group * outDim + j]
    mad.lo.u32 %r13, %r12, %r4, %r3;
    mul.wide.u32 %rd10, %r13, 4;
    add.u64 %rd10, %rd2, %rd10;
    ld.global.f32 %f11, [%rd10];  // cached scale

    // Unrolled: process all 8 values from the packed int32
    // bit 0: i = i_base + 0
    and.b32 %r14, %r10, 0xF;
    add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;
    // Load x[i_base+0] from shared memory
    mul.wide.u32 %rd11, %r11, 4;
    add.u64 %rd11, %rd6, %rd11;
    ld.shared.f32 %f2, [%rd11];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    // bit 1
    shr.u32 %r14, %r10, 4;
    and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;
    ld.shared.f32 %f2, [%rd11+4];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    // bit 2
    shr.u32 %r14, %r10, 8;
    and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;
    ld.shared.f32 %f2, [%rd11+8];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    // bit 3
    shr.u32 %r14, %r10, 12;
    and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;
    ld.shared.f32 %f2, [%rd11+12];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    // bit 4
    shr.u32 %r14, %r10, 16;
    and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;
    ld.shared.f32 %f2, [%rd11+16];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    // bit 5
    shr.u32 %r14, %r10, 20;
    and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;
    ld.shared.f32 %f2, [%rd11+20];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    // bit 6
    shr.u32 %r14, %r10, 24;
    and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;
    ld.shared.f32 %f2, [%rd11+24];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    // bit 7
    shr.u32 %r14, %r10, 28;
    and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;
    ld.shared.f32 %f2, [%rd11+28];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    add.u32 %r8, %r8, 1;
    bra pack_loop;
pack_done:

    // Store result
    ld.param.u64 %rd12, [param_out];
    mul.wide.u32 %rd13, %r3, 4;
    add.u64 %rd13, %rd12, %rd13;
    st.global.f32 [%rd13], %f10;

done:
    ret;
}
`
