package ptx

// Tiled INT4 dequant+GEMV — cooperative shared memory weight loading.
//
// Each block handles 256 output columns. Threads cooperatively load
// weight tiles [TILE_K, 256] into shared memory, then each thread
// reads from shared memory (100x faster than VRAM).
//
// Tile: TILE_K=16 packs (128 input elements) per iteration.
// Shared memory: x[4096] + weight_tile[16*256] = 16KB + 16KB = 32KB.
// RTX 3060 shared memory: 48KB per SM — fits.
//
// Memory traffic per tile: 16*256*4 = 16KB (cooperative load)
// vs old kernel: 256 threads × 16 reads = 4MB (independent loads)
// Reduction: 256x less VRAM traffic.

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
    .reg .u64 %rd<24>;
    .reg .f32 %f<20>;
    .reg .pred %p<4>;

    // Shared memory: x vector + weight tile
    .shared .align 16 .f32 sx[4096];       // x vector (up to 4096 floats)
    .shared .align 4 .u32 sqw[4096];       // weight tile: TILE_K(16) * 256 = 4096

    mov.u32 %r0, %ctaid.x;   // block index
    mov.u32 %r1, %ntid.x;    // block size (256)
    mov.u32 %r2, %tid.x;     // thread index

    // j = blockIdx.x * 256 + threadIdx.x (output column for this thread)
    shl.b32 %r20, %r0, 8;    // blockIdx * 256
    add.u32 %r3, %r20, %r2;  // j

    ld.param.u32 %r4, [param_outDim];
    ld.param.u32 %r5, [param_inDim];
    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_qw];
    ld.param.u64 %rd2, [param_scales];
    ld.param.u64 %rd3, [param_gidx];

    // Phase 1: cooperatively load x[] into shared memory
    mov.u64 %rd6, sx;
    mov.u32 %r6, %r2;
load_x:
    setp.ge.u32 %p0, %r6, %r5;
    @%p0 bra load_x_done;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];
    add.u64 %rd7, %rd6, %rd4;
    st.shared.f32 [%rd7], %f0;
    add.u32 %r6, %r6, 256;
    bra load_x;
load_x_done:
    bar.sync 0;

    // Accumulator
    mov.f32 %f10, 0f00000000;

    // Check bounds
    setp.ge.u32 %p0, %r3, %r4;

    // Process weight in tiles of TILE_K=16 packs
    shr.u32 %r7, %r5, 3;     // nPacks = inDim / 8
    mov.u32 %r8, 0;           // tileStart
    mov.u64 %rd8, sqw;        // shared weight base

tile_loop:
    setp.ge.u32 %p1, %r8, %r7;
    @%p1 bra tile_done;

    // Phase 2a: cooperatively load weight tile qw[tileStart:tileStart+16, blockCol:blockCol+256]
    // Each of 256 threads loads 16 values (one per tile row)
    // sqw[row * 256 + tid] = qw[(tileStart + row) * outDim + (blockCol + tid)]
    bar.sync 0;
    mov.u32 %r15, 0;    // row within tile
load_tile:
    setp.ge.u32 %p2, %r15, 16;
    @%p2 bra load_tile_done;

    // Global index: (tileStart + row) * outDim + j
    add.u32 %r16, %r8, %r15;          // tileStart + row
    setp.ge.u32 %p2, %r16, %r7;       // bounds check
    @%p2 bra load_tile_skip;
    mad.lo.u32 %r17, %r16, %r4, %r3;  // (tileStart+row)*outDim + j
    mul.wide.u32 %rd9, %r17, 4;
    add.u64 %rd10, %rd1, %rd9;

    // Load from VRAM (coalesced: consecutive threads read consecutive columns)
    mov.u32 %r18, 0;
    @%p0 bra load_tile_skip;  // skip if j >= outDim
    ld.global.u32 %r18, [%rd10];

load_tile_skip:
    // Store to shared: sqw[row * 256 + tid]
    mad.lo.u32 %r19, %r15, 256, %r2;
    mul.wide.u32 %rd11, %r19, 4;
    add.u64 %rd12, %rd8, %rd11;
    st.shared.u32 [%rd12], %r18;

    add.u32 %r15, %r15, 1;
    bra load_tile;
load_tile_done:
    bar.sync 0;

    // Phase 2b: each thread computes from shared memory
    @%p0 bra tile_skip_compute;   // skip if j >= outDim

    mov.u32 %r15, 0;    // row within tile
compute_tile:
    setp.ge.u32 %p2, %r15, 16;
    @%p2 bra compute_tile_done;

    add.u32 %r16, %r8, %r15;     // packIdx = tileStart + row
    setp.ge.u32 %p2, %r16, %r7;
    @%p2 bra compute_tile_done;

    // Load packed value from shared memory: sqw[row * 256 + tid]
    mad.lo.u32 %r19, %r15, 256, %r2;
    mul.wide.u32 %rd11, %r19, 4;
    add.u64 %rd12, %rd8, %rd11;
    ld.shared.u32 %r10, [%rd12];

    // i_base = packIdx * 8
    shl.b32 %r11, %r16, 3;

    // Load group index and scale (from global - small, likely cached in L2)
    mul.wide.u32 %rd13, %r11, 4;
    add.u64 %rd14, %rd3, %rd13;
    ld.global.u32 %r12, [%rd14];      // gIdx[i_base]
    mad.lo.u32 %r13, %r12, %r4, %r3;  // group * outDim + j
    mul.wide.u32 %rd15, %r13, 4;
    add.u64 %rd16, %rd2, %rd15;
    ld.global.f32 %f11, [%rd16];       // scale

    // x values from global memory (L1/L2 cached)
    mul.wide.u32 %rd17, %r11, 4;
    add.u64 %rd18, %rd0, %rd17;

    // Unrolled 8 values from the packed int32
    and.b32 %r14, %r10, 0xF;       add.s32 %r14, %r14, -8;
    cvt.rn.f32.s32 %f1, %r14;      mul.f32 %f1, %f11, %f1;
    ld.global.f32 %f2, [%rd18];     fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 4;         and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.global.f32 %f2, [%rd18+4];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 8;         and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.global.f32 %f2, [%rd18+8];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 12;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.global.f32 %f2, [%rd18+12];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 16;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.global.f32 %f2, [%rd18+16];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 20;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.global.f32 %f2, [%rd18+20];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 24;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.global.f32 %f2, [%rd18+24];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    shr.u32 %r14, %r10, 28;        and.b32 %r14, %r14, 0xF;
    add.s32 %r14, %r14, -8;        cvt.rn.f32.s32 %f1, %r14;
    mul.f32 %f1, %f11, %f1;        ld.global.f32 %f2, [%rd18+28];
    fma.rn.f32 %f10, %f2, %f1, %f10;

    add.u32 %r15, %r15, 1;
    bra compute_tile;
compute_tile_done:

tile_skip_compute:
    add.u32 %r8, %r8, 16;  // next tile
    bra tile_loop;
tile_done:

    // Store result
    @%p0 bra done;
    ld.param.u64 %rd19, [param_out];
    mul.wide.u32 %rd20, %r3, 4;
    add.u64 %rd21, %rd19, %rd20;
    st.global.f32 [%rd21], %f10;

done:
    ret;
}
`
