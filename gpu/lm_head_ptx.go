package gpu

// LM head GEMV: optimized kernel for [vocab × h] × [h] → [vocab]
//
// Each block computes one output element (one row's dot product).
// 256 threads per block, shared memory tree reduction.
// This is much faster than SGEMM for N=1 (vector) cases.

import (
	"unsafe"
)

// DevLMHead computes logits[vocab] = W[vocab,h] · x[h]
// Uses a dedicated kernel optimized for large M (vocab) and small N (1).
func DevLMHead(logits, x, W *DevBuf, vocab, h int) {
	if !kernelsLoaded || fnLMHead == 0 {
		DevGemv(logits, x, W, vocab, h)
		return
	}
	EnsureContext()
	x.ToGPU()
	W.ToGPU()
	logits.EnsureGPU()

	if x.gpu == nil || W.gpu == nil || logits.gpu == nil {
		DevGemv(logits, x, W, vocab, h)
		return
	}

	v := uint32(vocab)
	dim := uint32(h)

	// Grid: vocab may exceed 65535 max blocks in x. Use 2D grid.
	gridX := uint32(vocab)
	gridY := uint32(1)
	if gridX > 65535 {
		gridY = (gridX + 65534) / 65535
		gridX = 65535
	}

	LaunchKernel(fnLMHead, gridX, gridY, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&W.gpu.Ptr),
		unsafe.Pointer(&x.gpu.Ptr),
		unsafe.Pointer(&logits.gpu.Ptr),
		unsafe.Pointer(&v),
		unsafe.Pointer(&dim))
	logits.dev = GPU_DEVICE
}

var fnLMHead CUfunction

// LMHeadPTX: each block computes one output element.
// Block i computes: logits[i] = sum_k(W[i*h + k] * x[k])
// 256 threads stride over h dimension with shared memory reduction.
var LMHeadPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry lm_head_gemv(
    .param .u64 W,         // [vocab, h] f32
    .param .u64 x,         // [h] f32
    .param .u64 out,       // [vocab] f32
    .param .u32 vocab,
    .param .u32 h
) {
    .reg .u32 %r<16>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<8>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    // row = blockIdx.y * 65535 + blockIdx.x
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r8, %ctaid.y;
    mad.lo.u32 %r0, %r8, 65535, %r0;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [vocab];
    ld.param.u32 %r3, [h];

    // Bounds check
    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;

    // Load base pointers
    ld.param.u64 %rd0, [W];
    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [out];

    // W_row = W + row * h
    mul.lo.u32 %r4, %r0, %r3;
    mul.wide.u32 %rd3, %r4, 4;
    add.u64 %rd0, %rd0, %rd3;

    // Partial sum: stride by 256
    mov.f32 %f0, 0f00000000;
    mov.u32 %r5, %r1;

loop:
    setp.ge.u32 %p, %r5, %r3;
    @%p bra reduce;

    // Load W[row, k] and x[k]
    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];
    add.u64 %rd6, %rd1, %rd4;
    ld.global.f32 %f2, [%rd6];

    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r5, %r5, 256;
    bra loop;

reduce:
    // Store in shared memory
    mov.u64 %rd7, sdata;
    mul.wide.u32 %rd8, %r1, 4;
    add.u64 %rd9, %rd7, %rd8;
    st.shared.f32 [%rd9], %f0;
    bar.sync 0;

    // Tree reduction
    mov.u32 %r6, 128;
red_loop:
    setp.lt.u32 %p, %r6, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r1, %r6;
    @%p bra red_skip;

    add.u32 %r7, %r1, %r6;
    mul.wide.u32 %rd10, %r7, 4;
    add.u64 %rd11, %rd7, %rd10;
    ld.shared.f32 %f3, [%rd11];
    ld.shared.f32 %f4, [%rd9];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd9], %f4;

red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra red_loop;

red_done:
    // Thread 0 writes output
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.f32 %f5, [sdata];
    mul.wide.u32 %rd4, %r0, 4;
    add.u64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f5;

done:
    ret;
}
`
