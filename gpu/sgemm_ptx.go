package gpu

// PTX SGEMM kernel for NVIDIA GPUs (Ampere SM86+).
// Embedded as a Go string — compiled at runtime by the GPU driver.
//
// This is a tiled 16×16 SGEMM using shared memory.
// For RTX 3060 (28 SMs, 128 CUDA cores/SM = 3584 cores):
//   Theoretical: 12.7 TFLOPS FP32
//   Practical SGEMM: ~8-10 TFLOPS with this simple kernel

const SgemmPTX = `
.version 7.0
.target sm_80
.address_size 64

// sgemm_nn: C[M,N] = alpha * A[M,K] * B[K,N]
// Row-major, tiled 16x16 with shared memory
.visible .entry sgemm_nn(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_M,
    .param .u32 param_N,
    .param .u32 param_K,
    .param .f32 param_alpha
) {
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<16>;
    .reg .pred %p<4>;

    // Shared memory tiles: 16x16 floats each
    .shared .align 4 .f32 sA[256];  // 16x16
    .shared .align 4 .f32 sB[256];  // 16x16

    // Thread/block indices
    ld.param.u64 %rd0, [param_A];
    ld.param.u64 %rd1, [param_B];
    ld.param.u64 %rd2, [param_C];
    ld.param.u32 %r0, [param_M];    // M
    ld.param.u32 %r1, [param_N];    // N
    ld.param.u32 %r2, [param_K];    // K
    ld.param.f32 %f0, [param_alpha]; // alpha

    // Global row/col
    mov.u32 %r3, %ctaid.y;          // block row
    mov.u32 %r4, %ctaid.x;          // block col
    mov.u32 %r5, %tid.y;            // thread row in block (0-15)
    mov.u32 %r6, %tid.x;            // thread col in block (0-15)

    // Global row = blockRow * 16 + threadRow
    mad.lo.u32 %r7, %r3, 16, %r5;   // globalRow
    mad.lo.u32 %r8, %r4, 16, %r6;   // globalCol

    // Accumulator
    mov.f32 %f1, 0.0;               // sum = 0

    // Number of K-tiles
    add.u32 %r10, %r2, 15;
    shr.u32 %r10, %r10, 4;          // numTiles = (K+15)/16

    mov.u32 %r11, 0;                // tile = 0

TILE_LOOP:
    setp.ge.u32 %p0, %r11, %r10;
    @%p0 bra TILE_DONE;

    // Tile offset in K dimension
    shl.b32 %r12, %r11, 4;          // tileK = tile * 16

    // Load A tile: sA[threadRow][threadCol] = A[globalRow][tileK + threadCol]
    add.u32 %r13, %r12, %r6;        // kIdx = tileK + threadCol
    setp.lt.u32 %p1, %r7, %r0;      // globalRow < M
    setp.lt.u32 %p2, %r13, %r2;     // kIdx < K
    and.pred %p1, %p1, %p2;

    // sA offset: threadRow * 16 + threadCol
    mad.lo.u32 %r14, %r5, 16, %r6;
    mul.wide.u32 %rd3, %r14, 4;
    mov.u64 %rd4, sA;
    add.u64 %rd3, %rd4, %rd3;

    // A offset: globalRow * K + kIdx
    mad.lo.u32 %r15, %r7, %r2, %r13;
    mul.wide.u32 %rd5, %r15, 4;
    add.u64 %rd5, %rd0, %rd5;

    mov.f32 %f2, 0.0;
    @%p1 ld.global.f32 %f2, [%rd5];
    st.shared.f32 [%rd3], %f2;

    // Load B tile: sB[threadRow][threadCol] = B[tileK + threadRow][globalCol]
    add.u32 %r16, %r12, %r5;        // kIdx = tileK + threadRow
    setp.lt.u32 %p1, %r16, %r2;     // kIdx < K
    setp.lt.u32 %p2, %r8, %r1;      // globalCol < N
    and.pred %p1, %p1, %p2;

    // sB offset: threadRow * 16 + threadCol
    mov.u64 %rd6, sB;
    add.u64 %rd7, %rd6, %rd3;
    sub.u64 %rd7, %rd7, %rd4;       // reuse same local offset

    // B offset: (tileK + threadRow) * N + globalCol
    mad.lo.u32 %r17, %r16, %r1, %r8;
    mul.wide.u32 %rd8, %r17, 4;
    add.u64 %rd8, %rd1, %rd8;

    mov.f32 %f3, 0.0;
    @%p1 ld.global.f32 %f3, [%rd8];
    st.shared.f32 [%rd7], %f3;

    bar.sync 0;

    // Dot product over shared tile
    mov.u32 %r18, 0;                // p = 0
DOT_LOOP:
    setp.ge.u32 %p1, %r18, 16;
    @%p1 bra DOT_DONE;

    // sA[threadRow][p]
    mad.lo.u32 %r19, %r5, 16, %r18;
    mul.wide.u32 %rd9, %r19, 4;
    add.u64 %rd9, %rd4, %rd9;
    ld.shared.f32 %f4, [%rd9];

    // sB[p][threadCol]
    mad.lo.u32 %r20, %r18, 16, %r6;
    mul.wide.u32 %rd10, %r20, 4;
    add.u64 %rd10, %rd6, %rd10;
    ld.shared.f32 %f5, [%rd10];

    fma.rn.f32 %f1, %f4, %f5, %f1;

    add.u32 %r18, %r18, 1;
    bra DOT_LOOP;

DOT_DONE:
    bar.sync 0;

    add.u32 %r11, %r11, 1;
    bra TILE_LOOP;

TILE_DONE:
    // Write C[globalRow][globalCol] = alpha * sum
    setp.lt.u32 %p1, %r7, %r0;
    setp.lt.u32 %p2, %r8, %r1;
    and.pred %p1, %p1, %p2;

    mul.f32 %f1, %f1, %f0;          // sum *= alpha

    mad.lo.u32 %r21, %r7, %r1, %r8;
    mul.wide.u32 %rd11, %r21, 4;
    add.u64 %rd11, %rd2, %rd11;

    @%p1 st.global.f32 [%rd11], %f1;

    ret;
}
`
