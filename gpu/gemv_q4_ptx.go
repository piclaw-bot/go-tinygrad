package gpu

// Fused INT4 dequant + GEMV PTX kernel.
// Unpacks 4-bit weights, applies scale, and computes dot product in one pass.
// This keeps INT4 weights in VRAM (3.3GB for 7B) instead of F32 (24GB).
//
// GPTQ symmetric layout:
//   qweight[inDim/8, outDim] — 8 x 4-bit packed per int32
//   scales[numGroups, outDim] — F32 scale per group
//   gIdx[inDim] — group index per input row
//   Zero point = 8 (symmetric)
//
// Each thread computes one output element: out[j] = sum_i x[i] * scale[g[i],j] * (qw[i,j] - 8)

const GemvQ4PTX = `.version 7.0
.target sm_80
.address_size 64

// gemv_q4sym: out[outDim] = x[inDim] @ dequant(qweight[inDim/8, outDim])
// Args: x, qweight, scales, gIdx, out, inDim, outDim, numGroups
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
    .reg .u32 %r<20>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<8>;
    .reg .pred %p;

    // j = blockIdx.x * blockDim.x + threadIdx.x (output column)
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;  // j

    ld.param.u32 %r4, [param_outDim];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_qw];
    ld.param.u64 %rd2, [param_scales];
    ld.param.u64 %rd3, [param_gidx];
    ld.param.u32 %r5, [param_inDim];

    // Accumulator
    mov.f32 %f0, 0f00000000;

    // Loop over input dimension in packs of 8
    mov.u32 %r6, 0;  // packIdx
    shr.u32 %r7, %r5, 3;  // nPacks = inDim / 8

pack_loop:
    setp.ge.u32 %p, %r6, %r7;
    @%p bra pack_done;

    // Load packed int32: qw = qweight[packIdx * outDim + j]
    mad.lo.u32 %r8, %r6, %r4, %r3;  // packIdx * outDim + j
    mul.wide.u32 %rd4, %r8, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.u32 %r9, [%rd4];  // packed 8x4-bit

    // Process 8 values from this pack
    mov.u32 %r10, 0;  // bit (0..7)
bit_loop:
    setp.ge.u32 %p, %r10, 8;
    @%p bra bit_done;

    // i = packIdx * 8 + bit
    shl.b32 %r11, %r6, 3;
    add.u32 %r11, %r11, %r10;  // i

    // Extract 4-bit value: (packed >> (bit*4)) & 0xF
    shl.b32 %r12, %r10, 2;  // bit * 4
    shr.u32 %r13, %r9, %r12;
    and.b32 %r13, %r13, 0xF;  // qw_val (0..15)

    // Load x[i]
    mul.wide.u32 %rd5, %r11, 4;
    add.u64 %rd5, %rd0, %rd5;
    ld.global.f32 %f1, [%rd5];

    // Load gIdx[i] → group
    mul.wide.u32 %rd6, %r11, 4;
    add.u64 %rd6, %rd3, %rd6;
    ld.global.u32 %r14, [%rd6];  // group

    // Load scale[group * outDim + j]
    mad.lo.u32 %r15, %r14, %r4, %r3;
    mul.wide.u32 %rd7, %r15, 4;
    add.u64 %rd7, %rd2, %rd7;
    ld.global.f32 %f2, [%rd7];

    // dequant: w = scale * (qw_val - 8)
    add.s32 %r16, %r13, -8;  // qw_val - 8 (signed)
    cvt.rn.f32.s32 %f3, %r16;
    mul.f32 %f4, %f2, %f3;  // scale * (qw - 8)

    // accumulate: sum += x[i] * w
    fma.rn.f32 %f0, %f1, %f4, %f0;

    add.u32 %r10, %r10, 1;
    bra bit_loop;
bit_done:

    add.u32 %r6, %r6, 1;
    bra pack_loop;
pack_done:

    // Store result
    ld.param.u64 %rd8, [param_out];
    mul.wide.u32 %rd9, %r3, 4;
    add.u64 %rd9, %rd8, %rd9;
    st.global.f32 [%rd9], %f0;

done:
    ret;
}
`
