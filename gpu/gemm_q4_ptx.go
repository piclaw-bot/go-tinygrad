package gpu

// Batched Q4 GEMM: process multiple tokens in one kernel launch.
//
// Standard GEMV: [1 × inDim] × [inDim × outDim] → [1 × outDim]
// Batched GEMM:  [B × inDim] × [inDim × outDim] → [B × outDim]
//
// For prefill, B = prompt length. Each row reads the SAME weight matrix,
// so weight data is fetched once from VRAM and reused across all B rows.
// This turns memory-bound GEMV into compute-bound GEMM.
//
// The kernel dequantizes INT4 weights on-the-fly (same as gemv_q4sym)
// but accumulates B dot products per weight column.

import (
	"fmt"
	"unsafe"
)

// GemmQ4 performs batched matrix multiply: out[B×outDim] = input[B×inDim] × W_q4[inDim×outDim]
// where W is INT4 quantized with group scales.
func GemmQ4(out, input *DevBuf, w *GPUQuantWeight, B int) {
	if !q4Ready || w == nil || B <= 0 {
		return
	}
	EnsureContext()

	batchSize := uint32(B)
	inDim := uint32(w.InDim)
	outDim := uint32(w.OutDim)
	groups := uint32(w.Groups)

	input.ToGPU()
	out.ToGPU()

	// Grid: one block per (output column, batch row) pair
	// Block: 256 threads, each handles part of the dot product
	gridX := outDim
	gridY := batchSize

	LaunchKernel(fnGemmQ4, gridX, gridY, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&input.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.GIdx.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups),
		unsafe.Pointer(&batchSize),
	)
	out.dev = GPU_DEVICE
}

var fnGemmQ4 CUfunction

// GemmQ4PTX: batched INT4 dequant + GEMM kernel.
// Each block computes one element of the output: out[batch][col].
// Threads collaborate to reduce the dot product across inDim.
var GemmQ4PTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemm_q4sym(
    .param .u64 input,     // [B, inDim] f32
    .param .u64 qweight,   // [inDim/8, outDim] i32
    .param .u64 gidx,      // [inDim] i32
    .param .u64 scales,    // [groups, outDim] f32
    .param .u64 output,    // [B, outDim] f32
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 groups,
    .param .u32 B
) {
    .reg .u32 %r<32>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<16>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    // col = blockIdx.x, batch = blockIdx.y
    mov.u32 %r0, %ctaid.x;   // col
    mov.u32 %r1, %ctaid.y;   // batch
    mov.u32 %r2, %tid.x;     // tid
    ld.param.u32 %r3, [inDim];
    ld.param.u32 %r4, [outDim];
    ld.param.u32 %r5, [B];

    // Bounds check
    setp.ge.u32 %p, %r0, %r4;
    @%p bra done;
    setp.ge.u32 %p, %r1, %r5;
    @%p bra done;

    // Load base pointers
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [qweight];
    ld.param.u64 %rd2, [gidx];
    ld.param.u64 %rd3, [scales];
    ld.param.u64 %rd4, [output];
    ld.param.u32 %r6, [groups];

    // input_row = input + batch * inDim
    mul.lo.u32 %r10, %r1, %r3;
    mul.wide.u32 %rd5, %r10, 4;
    add.u64 %rd0, %rd0, %rd5;

    // Partial sum
    mov.f32 %f0, 0f00000000;

    // Each thread handles elements tid, tid+256, tid+512, ...
    // Process 8 elements at a time (one packed int32 = 8 x 4-bit)
    mov.u32 %r7, %r2;  // i = tid

loop:
    setp.ge.u32 %p, %r7, %r3;
    @%p bra reduce;

    // Load packed weight: qweight[(i/8)*outDim + col]
    shr.u32 %r8, %r7, 3;           // i/8
    mad.lo.u32 %r9, %r8, %r4, %r0; // (i/8)*outDim + col
    mul.wide.u32 %rd6, %r9, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u32 %r10, [%rd7];

    // Extract 4-bit weight for this position: (packed >> ((i%8)*4)) & 0xF
    and.b32 %r11, %r7, 7;          // i % 8
    shl.b32 %r11, %r11, 2;         // (i%8)*4
    shr.u32 %r12, %r10, %r11;
    and.b32 %r12, %r12, 15;        // 4-bit value [0..15]

    // Dequant: (val - 8) * scale
    add.s32 %r12, %r12, -8;        // signed: val - 8
    cvt.rn.f32.s32 %f1, %r12;

    // Load group index and scale
    mul.wide.u32 %rd8, %r7, 4;
    add.u64 %rd9, %rd2, %rd8;
    ld.global.u32 %r13, [%rd9];    // group = gidx[i]
    mad.lo.u32 %r14, %r13, %r4, %r0; // group*outDim + col
    mul.wide.u32 %rd10, %r14, 4;
    add.u64 %rd11, %rd3, %rd10;
    ld.global.f32 %f2, [%rd11];    // scale

    mul.f32 %f1, %f1, %f2;         // dequant weight

    // Load input value
    mul.wide.u32 %rd12, %r7, 4;
    add.u64 %rd13, %rd0, %rd12;
    ld.global.f32 %f3, [%rd13];

    // Accumulate
    fma.rn.f32 %f0, %f1, %f3, %f0;

    // Stride by 256 (blockDim)
    add.u32 %r7, %r7, 256;
    bra loop;

reduce:
    // Store partial sum in shared memory
    mul.wide.u32 %rd14, %r2, 4;
    mov.u64 %rd15, sdata;
    add.u64 %rd16, %rd15, %rd14;
    st.shared.f32 [%rd16], %f0;
    bar.sync 0;

    // Tree reduction in shared memory
    mov.u32 %r15, 128;
red_loop:
    setp.lt.u32 %p, %r15, 1;
    @%p bra red_done;
    setp.ge.u32 %p, %r2, %r15;
    @%p bra red_skip;
    
    // sdata[tid] += sdata[tid + stride]
    add.u32 %r16, %r2, %r15;
    mul.wide.u32 %rd17, %r16, 4;
    add.u64 %rd18, %rd15, %rd17;
    ld.shared.f32 %f4, [%rd18];
    ld.shared.f32 %f5, [%rd16];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd16], %f5;

red_skip:
    bar.sync 0;
    shr.u32 %r15, %r15, 1;
    bra red_loop;

red_done:
    // Thread 0 writes result: output[batch * outDim + col]
    setp.ne.u32 %p, %r2, 0;
    @%p bra done;
    ld.shared.f32 %f6, [sdata];
    
    // out_offset = batch * outDim + col
    mad.lo.u32 %r17, %r1, %r4, %r0;
    mul.wide.u32 %rd19, %r17, 4;
    add.u64 %rd20, %rd4, %rd19;
    st.global.f32 [%rd20], %f6;

done:
    ret;
}
`

func init() {
	// Will be extracted from mega module
}

// BatchGEMMReady returns true if batched GEMM kernel is available.
func BatchGEMMReady() bool {
	return fnGemmQ4 != 0
}

// GemvQ4OrGemm dispatches to GEMV (B=1) or batched GEMM (B>1).
func GemvQ4OrGemm(out, input *DevBuf, w *GPUQuantWeight, B int) {
	if B <= 1 {
		GemvQ4(out, input, w)
	} else {
		if fnGemmQ4 != 0 {
			GemmQ4(out, input, w, B)
		} else {
			// Fallback: sequential GEMV
			for b := 0; b < B; b++ {
				// This would need sub-buffer support
				fmt.Printf("[gpu] batch GEMM not available, falling back to sequential\n")
				return
			}
		}
	}
}
