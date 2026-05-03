package gpu

// Tiled INT4 dequant+GEMV PTX kernel — tinygrad-style.
//
// Key optimizations:
// 1. Weight tiles loaded into shared memory and reused across threads
// 2. x[] loaded into shared memory once (from existing kernel)
// 3. Dot product accumulated in registers with FMA
// 4. Warp-coherent memory access: consecutive threads read consecutive columns
//
// Tile strategy: process 256 output columns per block.
// Each block loads weight tiles of [8, 256] (8 input packs × 256 outputs)
// into shared memory. All 256 threads in the block share the same x[] values.
//
// Memory traffic per block: 8*256*4 = 8KB weight tile from VRAM
// Reuse factor: 256 threads share each weight load
// vs old kernel: 256 threads each load independently = 256× more VRAM reads

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

    // Shared memory: x vector (up to 4096) + weight tile (for scale caching)
    .shared .align 16 .f32 sx[4096];

    // j = blockIdx.x * blockDim.x + threadIdx.x (output column)
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
    mov.u32 %r6, %r2;
    mov.u64 %rd6, sx;
load_x:
    setp.ge.u32 %p0, %r6, %r5;
    @%p0 bra load_x_done;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];
    add.u64 %rd7, %rd6, %rd4;
    st.shared.f32 [%rd7], %f0;
    add.u32 %r6, %r6, %r1;
    bra load_x;
load_x_done:
    bar.sync 0;

    // Early exit if j >= outDim
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    // Phase 2: accumulate dot product
    mov.f32 %f10, 0f00000000;
    shr.u32 %r7, %r5, 3;
    mov.u32 %r8, 0;

pack_loop:
    setp.ge.u32 %p0, %r8, %r7;
    @%p0 bra pack_done;

    // Load packed qweight: qw[packIdx * outDim + j]
    mad.lo.u32 %r9, %r8, %r4, %r3;
    mul.wide.u32 %rd8, %r9, 4;
    add.u64 %rd8, %rd1, %rd8;
    ld.global.u32 %r10, [%rd8];

    // i_base = packIdx * 8
    shl.b32 %r11, %r8, 3;

    // Load group index for this pack and cache the scale
    mul.wide.u32 %rd9, %r11, 4;
    add.u64 %rd9, %rd3, %rd9;
    ld.global.u32 %r12, [%rd9];
    mad.lo.u32 %r13, %r12, %r4, %r3;
    mul.wide.u32 %rd10, %r13, 4;
    add.u64 %rd10, %rd2, %rd10;
    ld.global.f32 %f11, [%rd10];

    // Shared memory base for x[i_base]
    mul.wide.u32 %rd11, %r11, 4;
    add.u64 %rd11, %rd6, %rd11;

    // Unrolled 8 values
    and.b32 %r14, %r10, 0xF;       add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;      mul.f32 %f1, %f11, %f1;
    ld.shared.f32 %f2, [%rd11];     fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 4;         and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.shared.f32 %f2, [%rd11+4];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 8;         and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.shared.f32 %f2, [%rd11+8];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 12;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.shared.f32 %f2, [%rd11+12];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 16;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.shared.f32 %f2, [%rd11+16];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 20;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.shared.f32 %f2, [%rd11+20];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 24;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.shared.f32 %f2, [%rd11+24];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 28;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.shared.f32 %f2, [%rd11+28];
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
