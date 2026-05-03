package gpu

// MLX affine 4-bit GPU kernel: GEMV with MLX weight layout.
//
// MLX layout: weight[outDim, inDim/8] uint32, scales[outDim, numGroups], biases[outDim, numGroups]
// Dequant: val * scale + bias (per group)
//
// Each block computes one output row's dot product.
// 256 threads stride over inDim with shared memory reduction.

import (
	"fmt"
	"unsafe"
)

// GPUMLXWeight holds MLX quantized weight data on GPU.
type GPUMLXWeight struct {
	QWeight *Buffer // [outDim, inDim/8] packed uint32
	Scales  *Buffer // [outDim, numGroups] float32
	Biases  *Buffer // [outDim, numGroups] float32
	InDim   int
	OutDim  int
	Groups  int
	GroupSz int
}

// UploadMLXWeight uploads MLX quantized weight to GPU VRAM.
func UploadMLXWeight(weight []uint32, scales, biases []float32, inDim, outDim, groupSize int) (*GPUMLXWeight, error) {
	if !SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}
	EnsureContext()

	numGroups := inDim / groupSize
	packFactor := 8 // 4-bit, 8 values per uint32

	// Upload packed weights as int32 (same bits, just reinterpret)
	qwInt := make([]int32, len(weight))
	for i, v := range weight {
		qwInt[i] = int32(v)
	}
	qwBuf, err := Malloc(outDim * (inDim / packFactor))
	if err != nil {
		return nil, fmt.Errorf("alloc qweight: %w", err)
	}
	qwBuf.Upload(reinterpretI32asF32(qwInt))

	// Upload scales
	sBuf, err := Malloc(outDim * numGroups)
	if err != nil {
		return nil, fmt.Errorf("alloc scales: %w", err)
	}
	sBuf.Upload(scales)

	// Upload biases
	bBuf, err := Malloc(outDim * numGroups)
	if err != nil {
		return nil, fmt.Errorf("alloc biases: %w", err)
	}
	bBuf.Upload(biases)

	return &GPUMLXWeight{
		QWeight: qwBuf,
		Scales:  sBuf,
		Biases:  bBuf,
		InDim:   inDim,
		OutDim:  outDim,
		Groups:  numGroups,
		GroupSz: groupSize,
	}, nil
}

// reinterpretI32asF32 reinterprets []int32 as []float32 (same bits).
func reinterpretI32asF32(data []int32) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data))
}

// GemvMLX performs GPU GEMV with MLX quantized weights.
func GemvMLX(out *DevBuf, x *DevBuf, w *GPUMLXWeight) {
	if !q4Ready || fnMLXGemv == 0 || w == nil {
		return
	}
	EnsureContext()
	x.ToGPU()
	out.EnsureGPU()

	if x.gpu == nil || out.gpu == nil {
		return
	}

	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	groups := uint32(w.Groups)
	groupSz := uint32(w.GroupSz)

	LaunchKernel(fnMLXGemv, outDim, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&w.Biases.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups),
		unsafe.Pointer(&groupSz))
	out.dev = GPU_DEVICE
}

// GemmMLX performs batched GPU GEMM with MLX quantized weights.
// out[B, outDim] = input[B, inDim] × W_mlx[outDim, inDim]
func GemmMLX(out, input *DevBuf, w *GPUMLXWeight, B int) {
	if !q4Ready || fnMLXGemm == 0 || w == nil || B <= 0 {
		return
	}
	EnsureContext()
	input.ToGPU()
	out.EnsureGPU()

	if input.gpu == nil || out.gpu == nil {
		return
	}

	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	groups := uint32(w.Groups)
	groupSz := uint32(w.GroupSz)
	batchSize := uint32(B)

	LaunchKernel(fnMLXGemm, outDim, batchSize, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&input.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&w.Biases.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups),
		unsafe.Pointer(&groupSz),
		unsafe.Pointer(&batchSize))
	out.dev = GPU_DEVICE
}

var fnMLXGemv CUfunction
var fnMLXGemm CUfunction

// MLXGemvPTX: GEMV with MLX affine 4-bit weights.
// weight[outDim, inDim/8] uint32, scales/biases[outDim, numGroups]
// Each block computes one output element: out[row] = sum(dequant(W[row]) * x)
var MLXGemvPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_gemv(
    .param .u64 x,         // [inDim] f32
    .param .u64 qweight,   // [outDim, inDim/8] u32
    .param .u64 scales,    // [outDim, numGroups] f32
    .param .u64 biases,    // [outDim, numGroups] f32
    .param .u64 output,    // [outDim] f32
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize
) {
    .reg .u32 %r<24>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<12>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    // row = blockIdx.x (one block per output row)
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

    // qweight_row = qweight + row * (inDim/8) * 4
    shr.u32 %r6, %r3, 3;
    mul.lo.u32 %r7, %r0, %r6;
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd1, %rd1, %rd5;

    // scales_row = scales + row * numGroups * 4
    mul.lo.u32 %r8, %r0, %r4;
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd2, %rd2, %rd6;
    add.u64 %rd3, %rd3, %rd6;

    // Partial sum: each thread handles indices tid, tid+256, ...
    mov.f32 %f0, 0f00000000;
    mov.u32 %r9, %r1;

loop:
    setp.ge.u32 %p, %r9, %r3;
    @%p bra reduce;

    // Load packed weight: qweight[row, i/8]
    shr.u32 %r10, %r9, 3;
    mul.wide.u32 %rd7, %r10, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r11, [%rd8];

    // Extract 4-bit: (packed >> ((i%8)*4)) & 0xF
    and.b32 %r12, %r9, 7;
    shl.b32 %r12, %r12, 2;
    shr.u32 %r13, %r11, %r12;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f1, %r13;

    // Load scale and bias for this group: group = i / groupSize
    div.u32 %r14, %r9, %r5;
    mul.wide.u32 %rd9, %r14, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f2, [%rd10];
    add.u64 %rd11, %rd3, %rd9;
    ld.global.f32 %f3, [%rd11];

    // dequant: val * scale + bias
    fma.rn.f32 %f1, %f1, %f2, %f3;

    // Load x[i]
    mul.wide.u32 %rd12, %r9, 4;
    add.u64 %rd13, %rd0, %rd12;
    ld.global.f32 %f4, [%rd13];

    fma.rn.f32 %f0, %f1, %f4, %f0;

    add.u32 %r9, %r9, 256;
    bra loop;

reduce:
    mov.u64 %rd14, sdata;
    mul.wide.u32 %rd15, %r1, 4;
    add.u64 %rd14, %rd14, %rd15;
    st.shared.f32 [%rd14], %f0;
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
    ld.shared.f32 %f5, [%rd8];
    ld.shared.f32 %f6, [%rd14];
    add.f32 %f6, %f6, %f5;
    st.shared.f32 [%rd14], %f6;
red_skip:
    bar.sync 0;
    shr.u32 %r15, %r15, 1;
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

// MLXGemmPTX: batched GEMM with MLX weights (same kernel, 2D grid).
var MLXGemmPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_gemm(
    .param .u64 input,     // [B, inDim] f32
    .param .u64 qweight,   // [outDim, inDim/8] u32
    .param .u64 scales,    // [outDim, numGroups] f32
    .param .u64 biases,    // [outDim, numGroups] f32
    .param .u64 output,    // [B, outDim] f32
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize,
    .param .u32 B
) {
    .reg .u32 %r<24>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<12>;
    .reg .pred %p;
    .shared .align 4 .f32 sdata[256];

    // row = blockIdx.x, batch = blockIdx.y
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

    // input_row = input + batch * inDim
    mul.lo.u32 %r22, %r20, %r3;
    mul.wide.u32 %rd5, %r22, 4;
    add.u64 %rd0, %rd0, %rd5;

    // qweight_row = qweight + row * (inDim/8)
    shr.u32 %r6, %r3, 3;
    mul.lo.u32 %r7, %r0, %r6;
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd1, %rd1, %rd5;

    // scales/biases row offset
    mul.lo.u32 %r8, %r0, %r4;
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd2, %rd2, %rd6;
    add.u64 %rd3, %rd3, %rd6;

    mov.f32 %f0, 0f00000000;
    mov.u32 %r9, %r1;

loop:
    setp.ge.u32 %p, %r9, %r3;
    @%p bra reduce;

    shr.u32 %r10, %r9, 3;
    mul.wide.u32 %rd7, %r10, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r11, [%rd8];

    and.b32 %r12, %r9, 7;
    shl.b32 %r12, %r12, 2;
    shr.u32 %r13, %r11, %r12;
    and.b32 %r13, %r13, 15;
    cvt.rn.f32.u32 %f1, %r13;

    div.u32 %r14, %r9, %r5;
    mul.wide.u32 %rd9, %r14, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f2, [%rd10];
    add.u64 %rd11, %rd3, %rd9;
    ld.global.f32 %f3, [%rd11];

    fma.rn.f32 %f1, %f1, %f2, %f3;

    mul.wide.u32 %rd12, %r9, 4;
    add.u64 %rd13, %rd0, %rd12;
    ld.global.f32 %f4, [%rd13];

    fma.rn.f32 %f0, %f1, %f4, %f0;

    add.u32 %r9, %r9, 256;
    bra loop;

reduce:
    mov.u64 %rd14, sdata;
    mul.wide.u32 %rd15, %r1, 4;
    add.u64 %rd14, %rd14, %rd15;
    st.shared.f32 [%rd14], %f0;
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
    ld.shared.f32 %f5, [%rd8];
    ld.shared.f32 %f6, [%rd14];
    add.f32 %f6, %f6, %f5;
    st.shared.f32 [%rd14], %f6;
red_skip:
    bar.sync 0;
    shr.u32 %r15, %r15, 1;
    bra red_loop;

red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.f32 %f7, [sdata];
    // output[batch * outDim + row]
    mad.lo.u32 %r17, %r20, %r2, %r0;
    mul.wide.u32 %rd7, %r17, 4;
    add.u64 %rd8, %rd4, %rd7;
    st.global.f32 [%rd8], %f7;

done:
    ret;
}
`
