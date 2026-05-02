package gpu

// RoPE and Attention PTX kernels for fully GPU-resident forward pass.

// RoPE: apply rotary position embedding in-place on Q or K.
// Each thread handles one (cos,sin) pair for one head.
const RoPEPTX = `.version 7.0
.target sm_80
.address_size 64

// rope_apply: x[nHeads * headDim] updated in-place
// cos_sin[maxSeq * headDim] = interleaved [cos0,sin0, cos1,sin1, ...] per position
// Each thread handles one pair for one head
.visible .entry rope_apply(
    .param .u64 param_x,
    .param .u64 param_cos_sin,
    .param .u32 param_pos,
    .param .u32 param_nHeads,
    .param .u32 param_headDim
) {
    .reg .u32 %r<16>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<12>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_nHeads];
    ld.param.u32 %r5, [param_headDim];
    shr.u32 %r6, %r5, 1;

    mul.lo.u32 %r7, %r4, %r6;
    setp.ge.u32 %p, %r3, %r7;
    @%p bra done;

    div.u32 %r8, %r3, %r6;
    rem.u32 %r9, %r3, %r6;

    // idx = head * headDim + i * 2
    mul.lo.u32 %r10, %r8, %r5;
    shl.b32 %r11, %r9, 1;
    add.u32 %r10, %r10, %r11;

    // Load x[idx], x[idx+1]
    ld.param.u64 %rd0, [param_x];
    mul.wide.u32 %rd1, %r10, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f0, [%rd2];
    ld.global.f32 %f1, [%rd2+4];

    // Load precomputed cos, sin from cos_sin[pos * headDim + i*2], [pos * headDim + i*2 + 1]
    ld.param.u32 %r12, [param_pos];
    ld.param.u64 %rd3, [param_cos_sin];
    mul.lo.u32 %r13, %r12, %r5;
    add.u32 %r13, %r13, %r11;
    mul.wide.u32 %rd4, %r13, 4;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.f32 %f3, [%rd5];      // cos
    ld.global.f32 %f4, [%rd5+4];    // sin

    // Rotate
    mul.f32 %f5, %f0, %f3;
    mul.f32 %f6, %f1, %f4;
    sub.f32 %f7, %f5, %f6;

    mul.f32 %f8, %f0, %f4;
    mul.f32 %f9, %f1, %f3;
    add.f32 %f10, %f8, %f9;

    st.global.f32 [%rd2], %f7;
    st.global.f32 [%rd2+4], %f10;

done:
    ret;
}
`

// Attention: single-query attention against KV cache.
// For each head: score = softmax(Q*K^T / sqrt(d)) * V
// Launches one block per head, threads cooperate on the sequence dimension.
const AttentionPTX = `.version 7.0
.target sm_80
.address_size 64

// gqa_attention: out[nHeads*headDim] = attention(q, kCache, vCache)
// q[nHeads*headDim], kCache[seqLen*kvDim], vCache[seqLen*kvDim]
// One block per query head. Threads sweep the sequence.
// GQA: multiple query heads share one KV head (nHeads/nKVHeads ratio).
.visible .entry gqa_attention(
    .param .u64 param_q,
    .param .u64 param_k,
    .param .u64 param_v,
    .param .u64 param_out,
    .param .u32 param_seqLen,
    .param .u32 param_nHeads,
    .param .u32 param_nKVHeads,
    .param .u32 param_headDim
) {
    .reg .u32 %r<24>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<16>;
    .reg .pred %p<4>;

    // Shared memory for scores (max 2048 sequence length)
    .shared .align 4 .f32 scores[2048];

    // head = blockIdx.x (one block per query head)
    mov.u32 %r0, %ctaid.x;   // query head index
    mov.u32 %r1, %tid.x;     // thread within block

    ld.param.u32 %r2, [param_seqLen];
    ld.param.u32 %r3, [param_nHeads];
    ld.param.u32 %r4, [param_nKVHeads];
    ld.param.u32 %r5, [param_headDim];

    // kvHead = head / (nHeads / nKVHeads)
    div.u32 %r6, %r3, %r4;   // heads per KV head
    div.u32 %r7, %r0, %r6;   // kv head index

    // scale = 1/sqrt(headDim)
    cvt.rn.f32.u32 %f0, %r5;
    rsqrt.approx.f32 %f1, %f0; // 1/sqrt(headDim)

    ld.param.u64 %rd0, [param_q];
    ld.param.u64 %rd1, [param_k];
    ld.param.u64 %rd2, [param_v];

    // q_offset = head * headDim
    mul.lo.u32 %r8, %r0, %r5;

    // Phase 1: compute attention scores = Q * K^T / sqrt(d)
    // Each thread handles one or more sequence positions
    mov.u32 %r9, %r1; // seq pos
score_loop:
    setp.ge.u32 %p0, %r9, %r2;
    @%p0 bra score_done;

    // dot(q[head], k[seq, kvHead])
    // k_offset = seq * kvDim + kvHead * headDim
    // kvDim = nKVHeads * headDim
    mul.lo.u32 %r10, %r4, %r5;   // kvDim
    mul.lo.u32 %r11, %r9, %r10;  // seq * kvDim
    mul.lo.u32 %r12, %r7, %r5;   // kvHead * headDim
    add.u32 %r11, %r11, %r12;    // k_offset

    mov.f32 %f2, 0f00000000;     // dot product accumulator
    mov.u32 %r13, 0;             // d index
dot_loop:
    setp.ge.u32 %p1, %r13, %r5;
    @%p1 bra dot_done;

    // q[head*headDim + d]
    add.u32 %r14, %r8, %r13;
    mul.wide.u32 %rd3, %r14, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f3, [%rd4];

    // k[seq*kvDim + kvHead*headDim + d]
    add.u32 %r15, %r11, %r13;
    mul.wide.u32 %rd5, %r15, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.f32 %f4, [%rd6];

    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r13, %r13, 1;
    bra dot_loop;
dot_done:

    // score = dot * scale
    mul.f32 %f2, %f2, %f1;

    // Store in shared memory
    mul.wide.u32 %rd7, %r9, 4;
    mov.u64 %rd8, scores;
    add.u64 %rd7, %rd8, %rd7;
    st.shared.f32 [%rd7], %f2;

    add.u32 %r9, %r9, 256; // += blockDim
    bra score_loop;
score_done:
    bar.sync 0;

    // Phase 2: softmax (thread 0 does it - sequential but short for small seqLen)
    setp.ne.u32 %p0, %r1, 0;
    @%p0 bra softmax_done;

    // Find max
    mov.f32 %f5, 0fFF800000; // -inf
    mov.u32 %r9, 0;
max_loop:
    setp.ge.u32 %p1, %r9, %r2;
    @%p1 bra max_done;
    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd7, %rd8, %rd7;
    ld.shared.f32 %f6, [%rd7];
    max.f32 %f5, %f5, %f6;
    add.u32 %r9, %r9, 1;
    bra max_loop;
max_done:

    // exp and sum
    mov.f32 %f7, 0f00000000; // sum
    mov.u32 %r9, 0;
exp_loop:
    setp.ge.u32 %p1, %r9, %r2;
    @%p1 bra exp_done;
    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd7, %rd8, %rd7;
    ld.shared.f32 %f6, [%rd7];
    sub.f32 %f6, %f6, %f5;           // x - max
    mul.f32 %f6, %f6, 0f3FB8AA3B;    // * log2(e)
    ex2.approx.f32 %f6, %f6;         // exp(x-max)
    add.f32 %f7, %f7, %f6;
    st.shared.f32 [%rd7], %f6;       // store exp
    add.u32 %r9, %r9, 1;
    bra exp_loop;
exp_done:

    // Normalize
    rcp.rn.f32 %f7, %f7;         // 1/sum
    mov.u32 %r9, 0;
norm_loop:
    setp.ge.u32 %p1, %r9, %r2;
    @%p1 bra norm_done;
    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd7, %rd8, %rd7;
    ld.shared.f32 %f6, [%rd7];
    mul.f32 %f6, %f6, %f7;
    st.shared.f32 [%rd7], %f6;
    add.u32 %r9, %r9, 1;
    bra norm_loop;
norm_done:

softmax_done:
    bar.sync 0;

    // Phase 3: weighted sum of V
    // out[head*headDim + d] = sum_s scores[s] * v[s*kvDim + kvHead*headDim + d]
    // Each thread handles one or more d dimensions
    ld.param.u64 %rd9, [param_out];
    mul.lo.u32 %r10, %r4, %r5;   // kvDim

    mov.u32 %r13, %r1;           // d index
vsum_loop:
    setp.ge.u32 %p0, %r13, %r5;
    @%p0 bra vsum_done;

    mov.f32 %f8, 0f00000000;     // weighted sum
    mov.u32 %r9, 0;              // seq pos
vs_inner:
    setp.ge.u32 %p1, %r9, %r2;
    @%p1 bra vs_inner_done;

    // scores[s]
    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd7, %rd8, %rd7;
    ld.shared.f32 %f9, [%rd7];

    // v[s*kvDim + kvHead*headDim + d]
    mul.lo.u32 %r15, %r9, %r10;
    mul.lo.u32 %r16, %r7, %r5;
    add.u32 %r15, %r15, %r16;
    add.u32 %r15, %r15, %r13;
    mul.wide.u32 %rd10, %r15, 4;
    add.u64 %rd11, %rd2, %rd10;
    ld.global.f32 %f10, [%rd11];

    fma.rn.f32 %f8, %f9, %f10, %f8;
    add.u32 %r9, %r9, 1;
    bra vs_inner;
vs_inner_done:

    // Store out[head*headDim + d]
    add.u32 %r17, %r8, %r13;
    mul.wide.u32 %rd12, %r17, 4;
    add.u64 %rd13, %rd9, %rd12;
    st.global.f32 [%rd13], %f8;

    add.u32 %r13, %r13, 256;    // += blockDim
    bra vsum_loop;
vsum_done:
    ret;
}
`
