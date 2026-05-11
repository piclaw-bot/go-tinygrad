package gpu

// Device-agnostic compute buffer — tinygrad approach.
// Data lives on either CPU or GPU. Ops dispatch to the right backend.
// Transfers happen lazily when needed.

import (
	"fmt"
	"math"
	"sync"
	"unsafe"
)

// Device represents where data lives.
type Device int

const (
	CPU Device = iota
	GPU_DEVICE
)

// DevBuf is a device-agnostic buffer that can live on CPU or GPU.
type DevBuf struct {
	cpu    []float32 // CPU data (nil if GPU-only)
	gpu    *Buffer   // GPU data (nil if CPU-only)
	n      int       // number of float32 elements
	dev    Device    // current authoritative location
	ownGPU bool      // true if this DevBuf owns gpu and may free it
}

// Kernel function pointers (loaded once)
var (
	kernelsOnce      sync.Once
	kernelsLoaded    bool
	fnVecAdd         CUfunction
	fnVecMul         CUfunction
	fnVecScale       CUfunction
	fnToBF16F32      CUfunction
	fnVecSilu        CUfunction
	fnRmsNorm        CUfunction
	fnRmsNormNoScale CUfunction
	fnGELUTanhMul    CUfunction
)

func initKernels() { loadMegaModule() }

// NewDevBuf creates a CPU buffer.
func NewDevBuf(n int) *DevBuf {
	return &DevBuf{cpu: make([]float32, n), n: n, dev: CPU}
}

// NewDevBufFrom wraps existing CPU data.
func NewDevBufFrom(data []float32) *DevBuf {
	return &DevBuf{cpu: data, n: len(data), dev: CPU}
}

// ToGPU ensures data is on GPU. No-op if already there.
func (b *DevBuf) ToGPU() error {
	if b.gpu != nil {
		if b.dev == CPU && b.cpu != nil {
			// CPU data was modified — re-upload
			b.gpu.Upload(b.cpu)
		}
		b.dev = GPU_DEVICE
		return nil
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	var err error
	b.gpu, err = Malloc(b.n)
	if err != nil {
		return err
	}
	b.ownGPU = true
	if b.cpu != nil {
		b.gpu.Upload(b.cpu)
	}
	b.dev = GPU_DEVICE
	return nil
}

// ToCPU ensures data is on CPU. No-op if already there.
func (b *DevBuf) ToCPU() {
	if b.cpu == nil {
		b.cpu = make([]float32, b.n)
	}
	if b.gpu != nil && b.dev == GPU_DEVICE {
		b.gpu.Download(b.cpu)
	}
	b.dev = CPU
}

// Data returns CPU-side data (downloading from GPU if needed).
func (b *DevBuf) Data() []float32 {
	b.ToCPU()
	return b.cpu
}

// EnsureGPU ensures GPU buffer exists without uploading CPU data.
func (b *DevBuf) EnsureGPU() error {
	if b.gpu != nil {
		return nil
	}
	return b.ToGPU()
}

// GPUPtr returns the GPU buffer, uploading if needed.
func (b *DevBuf) GPUPtr() *Buffer {
	if b.gpu == nil {
		b.ToGPU()
	}
	return b.gpu
}

// Len returns element count.
func (b *DevBuf) Len() int { return b.n }

// OnGPU returns true if data is authoritatively on GPU.
func (b *DevBuf) OnGPU() bool { return b.dev == GPU_DEVICE && b.gpu != nil }

// tryGPU attempts to move buffers to GPU. Returns true if all succeeded.
func tryGPU(bufs ...*DevBuf) bool {
	for _, b := range bufs {
		if b.ToGPU() != nil || b.gpu == nil {
			return false
		}
	}
	return true
}

// --- Ops: dispatch to GPU if possible, CPU fallback ---

// Add: out = a + b (element-wise)
func DevAdd(out, a, b *DevBuf) {
	initKernels()
	n := a.n
	if kernelsLoaded && tryGPU(a, b, out) {
		a.ToGPU()
		b.ToGPU()
		out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnVecAdd, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&b.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr), unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	// CPU fallback
	a.ToCPU()
	b.ToCPU()
	out.ToCPU()
	for i := 0; i < n; i++ {
		out.cpu[i] = a.cpu[i] + b.cpu[i]
	}
}

// Mul: out = a * b (element-wise)
func DevMul(out, a, b *DevBuf) {
	initKernels()
	n := a.n
	if kernelsLoaded && tryGPU(a, b, out) {
		a.ToGPU()
		b.ToGPU()
		out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnVecMul, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&b.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr), unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	a.ToCPU()
	b.ToCPU()
	out.ToCPU()
	for i := 0; i < n; i++ {
		out.cpu[i] = a.cpu[i] * b.cpu[i]
	}
}

// Scale: out = a * scalar
func DevScale(out, a *DevBuf, s float32) {
	initKernels()
	n := a.n
	if kernelsLoaded && tryGPU(a, out) {
		a.ToGPU()
		out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnVecScale, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&s), unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	a.ToCPU()
	out.ToCPU()
	for i := 0; i < n; i++ {
		out.cpu[i] = a.cpu[i] * s
	}
}

// ToBF16: truncate float32 values in-place to BF16 precision.
func DevToBF16(x *DevBuf, n int) {
	initKernels()
	if n <= 0 {
		n = x.n
	}
	if kernelsLoaded && fnToBF16F32 != 0 && tryGPU(x) {
		x.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnToBF16F32, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&x.gpu.Ptr), unsafe.Pointer(&nn))
		x.dev = GPU_DEVICE
		return
	}
	x.ToCPU()
	if n > x.n {
		n = x.n
	}
	for i := 0; i < n; i++ {
		bits := math.Float32bits(x.cpu[i])
		bits &= 0xFFFF0000
		x.cpu[i] = math.Float32frombits(bits)
	}
}

// SiLU: out = a * sigmoid(a)
func DevSiLU(out, a *DevBuf) {
	initKernels()
	n := a.n
	if kernelsLoaded && tryGPU(a, out) {
		a.ToGPU()
		out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnVecSilu, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	a.ToCPU()
	out.ToCPU()
	for i := 0; i < n; i++ {
		x := a.cpu[i]
		out.cpu[i] = x / (1.0 + float32(math.Exp(float64(-x))))
	}
}

// RMSNorm: out = x * weight * rsqrt(mean(x^2) + eps)
func DevRMSNorm(out, x, weight *DevBuf, eps float32) {
	initKernels()
	n := weight.n // use weight size as canonical dimension (handles oversized buffers)
	if n > x.n {
		n = x.n
	}
	if kernelsLoaded && n <= 256*8192 {
		x.ToGPU()
		weight.ToGPU()
		out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnRmsNorm, 1, 1, 1, 256, 1, 1, 256*4,
			unsafe.Pointer(&x.gpu.Ptr), unsafe.Pointer(&weight.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&eps))
		out.dev = GPU_DEVICE
		return
	}
	// CPU fallback
	x.ToCPU()
	weight.ToCPU()
	out.ToCPU()
	var ss float32
	for _, v := range x.cpu[:n] {
		ss += v * v
	}
	ss = 1.0 / float32(math.Sqrt(float64(ss/float32(n)+eps)))
	for i := 0; i < n; i++ {
		out.cpu[i] = x.cpu[i] * ss * weight.cpu[i]
	}
}

// DevRMSNormNoScale: normalize x by RMS without weight. out = x / rms(x)
func DevRMSNormNoScale(out, x *DevBuf, eps float32) {
	initKernels()
	n := x.n
	if kernelsLoaded && fnRmsNormNoScale != 0 && n <= 256*8192 {
		x.ToGPU()
		out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnRmsNormNoScale, 1, 1, 1, 256, 1, 1, 256*4,
			unsafe.Pointer(&x.gpu.Ptr), unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&nn), unsafe.Pointer(&eps))
		out.dev = GPU_DEVICE
		return
	}
	// CPU fallback
	x.ToCPU()
	out.ToCPU()
	var ss float32
	for _, v := range x.cpu[:n] {
		ss += v * v
	}
	ss = 1.0 / float32(math.Sqrt(float64(ss/float32(n)+eps)))
	for i := 0; i < n; i++ {
		out.cpu[i] = x.cpu[i] * ss
	}
}

// Gemv: out[M] = W[M,K] * x[K] (matrix-vector multiply)
func DevGemv(out, x *DevBuf, W *DevBuf, M, K int) {
	if SgemmReady() {
		if x.ToGPU() == nil && W.ToGPU() == nil && out.ToGPU() == nil {
			Sgemm(M, 1, K, 1.0, W.gpu, x.gpu, out.gpu)
			out.dev = GPU_DEVICE
			return
		}
	}
	// CPU fallback: out[j] = dot(W[j,:], x)
	x.ToCPU()
	W.ToCPU()
	out.ToCPU()
	w := W.cpu
	xd := x.cpu
	for j := 0; j < M; j++ {
		sum := float32(0)
		row := w[j*K : (j+1)*K]
		for p := 0; p < K; p++ {
			sum += xd[p] * row[p]
		}
		out.cpu[j] = sum
	}
}

// Softmax in-place (CPU only for now — sequential reduction)
func DevSoftmax(x *DevBuf, n int) {
	x.ToCPU()
	d := x.cpu[:n]
	max := d[0]
	for _, v := range d[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range d {
		d[i] = float32(math.Exp(float64(v - max)))
		sum += d[i]
	}
	for i := range d {
		d[i] /= sum
	}
}

// Copy copies src data to dst (same device).
func DevCopy(dst, src *DevBuf) {
	if src.gpu != nil && dst.gpu != nil && src.n >= 2048 {
		src.ToGPU()
		dst.ToGPU()
		cuMemcpyDtoDAsync(dst.gpu.Ptr, src.gpu.Ptr, uint64(src.n*4), 0) // stream 0 = default
		dst.dev = GPU_DEVICE
		return
	}
	src.ToCPU()
	dst.ToCPU()
	copy(dst.cpu, src.cpu[:dst.n])
}

// MarkDirty marks CPU data as authoritative (will re-upload on next GPU access).
func (b *DevBuf) MarkDirty() {
	b.dev = CPU
}

// MarkOnGPU marks GPU data as authoritative after in-place GPU-side mutation.
func (b *DevBuf) MarkOnGPU() {
	b.dev = GPU_DEVICE
}

// Free releases owned GPU resources held by this buffer.
// CPU memory is left to Go; non-owning slice views do not free the parent pointer.
func (b *DevBuf) Free() {
	if b == nil {
		return
	}
	if b.gpu != nil && b.ownGPU {
		b.gpu.Free()
	}
	b.gpu = nil
	b.ownGPU = false
}

// GemvNN: out[N] = x[K] @ W[K,N] (W is pre-transposed, column-major for output)
// This is for the non-Large path where weights are pre-transposed.
func DevGemvNN(out, x *DevBuf, W *DevBuf, K, N int) {
	if SgemmReady() {
		if x.ToGPU() == nil && W.ToGPU() == nil && out.ToGPU() == nil {
			Sgemm(1, N, K, 1.0, x.gpu, W.gpu, out.gpu)
			out.dev = GPU_DEVICE
			return
		}
	}
	// CPU fallback
	x.ToCPU()
	W.ToCPU()
	out.ToCPU()
	xd := x.cpu
	w := W.cpu
	for j := 0; j < N; j++ {
		sum := float32(0)
		for p := 0; p < K; p++ {
			sum += xd[p] * w[p*N+j]
		}
		out.cpu[j] = sum
	}
}

// --- GPU RoPE + Attention ---

var (
	ropeOnce         sync.Once
	ropeFn           CUfunction
	ropePartialFn    CUfunction
	attnScoreFn      CUfunction
	softmaxRowsFn    CUfunction
	attnFn           CUfunction
	ropeReady        bool
	ropePartialReady bool
	attnScoreReady   bool
	softmaxRowsReady bool
	attnReady        bool
)

func initRoPEAttn() { loadMegaModule() }

// DevRoPE applies rotary position embedding on GPU (in-place).
// cosSin is a precomputed [maxSeq * headDim] buffer with interleaved cos,sin pairs.
func DevRoPE(x *DevBuf, cosSin *DevBuf, pos, nHeads, headDim int) {
	initRoPEAttn()
	if ropeReady && tryGPU(x, cosSin) {
		p := uint32(pos)
		nh := uint32(nHeads)
		hd := uint32(headDim)
		halfDim := nHeads * (headDim / 2)
		LaunchKernel(ropeFn, (uint32(halfDim)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&x.gpu.Ptr),
			unsafe.Pointer(&cosSin.gpu.Ptr),
			unsafe.Pointer(&p),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&hd))
		return
	}
	// CPU fallback in model code
}

// DevRoPEPartial applies partial rotary position embedding on GPU (in-place).
// cosSin is [maxSeq * rotHalf * 2] with interleaved cos,sin pairs.
func DevRoPEPartial(x *DevBuf, cosSin *DevBuf, pos, nHeads, headDim, rotHalf int) bool {
	initRoPEAttn()
	if ropePartialReady && tryGPU(x, cosSin) {
		p := uint32(pos)
		nh := uint32(nHeads)
		hd := uint32(headDim)
		rh := uint32(rotHalf)
		totalPairs := nHeads * rotHalf
		LaunchKernel(ropePartialFn, (uint32(totalPairs)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&x.gpu.Ptr),
			unsafe.Pointer(&cosSin.gpu.Ptr),
			unsafe.Pointer(&p),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&rh))
		return true
	}
	return false
}

// DevAttentionScores runs the score phase of GQA attention on GPU.
// out[nHeads*seqLen], q[nHeads*headDim], kCache[seqLen*kvDim]
func DevAttentionScores(out, q, kCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) bool {
	initRoPEAttn()
	if attnScoreReady && seqLen <= 2048 && tryGPU(out, q, kCache) {
		sl := uint32(seqLen)
		nh := uint32(nHeads)
		nkv := uint32(nKVHeads)
		hd := uint32(headDim)
		LaunchKernel(attnScoreFn, uint32(nHeads), 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&q.gpu.Ptr),
			unsafe.Pointer(&kCache.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&nkv),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&scale))
		out.dev = GPU_DEVICE
		return true
	}
	return false
}

// DevSoftmaxRows runs the softmax phase over contiguous score rows.
// in/out[nRows*seqLen], one block per row.
func DevSoftmaxRows(out, in *DevBuf, nRows, seqLen int) bool {
	initRoPEAttn()
	if softmaxRowsReady && seqLen <= 2048 && tryGPU(out, in) {
		sl := uint32(seqLen)
		LaunchKernel(softmaxRowsFn, uint32(nRows), 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&in.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl))
		out.dev = GPU_DEVICE
		return true
	}
	return false
}

// DevAttention runs GQA attention on GPU.
// q[nHeads*headDim], kCache/vCache[seqLen*kvDim], out[nHeads*headDim]
func DevAttention(out, q, kCache, vCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) {
	initRoPEAttn()
	if attnReady && seqLen <= 2048 && tryGPU(out, q, kCache, vCache) {
		sl := uint32(seqLen)
		nh := uint32(nHeads)
		nkv := uint32(nKVHeads)
		hd := uint32(headDim)
		// One block per query head, 256 threads per block
		LaunchKernel(attnFn, uint32(nHeads), 1, 1, 256, 1, 1, 2048*4,
			unsafe.Pointer(&q.gpu.Ptr),
			unsafe.Pointer(&kCache.gpu.Ptr),
			unsafe.Pointer(&vCache.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&nkv),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&scale))
		out.dev = GPU_DEVICE
		return
	}
	// CPU fallback in model code
}

// CopyDtoD wraps cuMemcpyDtoD for direct GPU→GPU copy.
func CopyDtoD(dst, src CUdeviceptr, bytes uint64) {
	EnsureContext()
	Sync() // ensure pending ops complete before copy
	cuMemcpyDtoD(dst, src, bytes)
}

// Fused SiLU*Mul
var (
	fusedSiLUMulOnce sync.Once
	fnFusedSiLUMul   CUfunction
	fusedSiLUMulOK   bool
)

// DevSiLUMul computes out = silu(a) * b in one kernel launch
func DevSiLUMul(out, a, b *DevBuf) {
	initKernels()
	n := a.n
	if fusedSiLUMulOK && tryGPU(a, b, out) {
		nn := uint32(n)
		LaunchKernel(fnFusedSiLUMul, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&b.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr), unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	// CPU fallback
	a.ToCPU()
	b.ToCPU()
	out.ToCPU()
	for i := 0; i < n; i++ {
		x := a.cpu[i]
		out.cpu[i] = x / (1.0 + float32(math.Exp(float64(-x)))) * b.cpu[i]
	}
}

// DevGELUTanhMul: gate[i] = gelu_tanh(gate[i]) * up[i] in-place
func DevGELUTanhMul(gate, up *DevBuf, n int) {
	initKernels()
	if kernelsLoaded && fnGELUTanhMul != 0 && tryGPU(gate, up) {
		nn := uint32(n)
		LaunchKernel(fnGELUTanhMul, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&gate.gpu.Ptr), unsafe.Pointer(&up.gpu.Ptr),
			unsafe.Pointer(&nn))
		gate.dev = GPU_DEVICE
		return
	}
	// CPU fallback
	gate.ToCPU()
	up.ToCPU()
	for i := 0; i < n; i++ {
		x := gate.cpu[i]
		// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
		x3 := x * x * x
		z := 0.7978845608 * (x + 0.044715*x3)
		tanh_z := float32(math.Tanh(float64(z)))
		gate.cpu[i] = 0.5 * x * (1 + tanh_z) * up.cpu[i]
	}
}

var (
	jitSiLUMul *CompiledKernel
	jitAdd     *CompiledKernel
)

// Slice returns a DevBuf view into a sub-range [offset:offset+n] of this buffer.
// The slice shares CPU memory with the parent. GPU pointer is offset accordingly.
// The caller must not outlive the parent buffer.
func (b *DevBuf) Slice(offset, n int) *DevBuf {
	s := &DevBuf{n: n, dev: b.dev, ownGPU: false}
	if b.cpu != nil && offset+n <= len(b.cpu) {
		s.cpu = b.cpu[offset : offset+n]
	}
	if b.gpu != nil {
		s.gpu = &Buffer{
			Ptr:  b.gpu.Ptr + CUdeviceptr(uint64(offset)*4),
			Size: n * 4,
		}
	}
	return s
}
