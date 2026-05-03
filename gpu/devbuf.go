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
	cpu  []float32  // CPU data (nil if GPU-only)
	gpu  *Buffer    // GPU data (nil if CPU-only)
	n    int        // number of float32 elements
	dev  Device     // current authoritative location
}

// Kernel function pointers (loaded once)
var (
	kernelsOnce   sync.Once
	kernelsLoaded bool
	fnVecAdd      CUfunction
	fnVecMul      CUfunction
	fnVecScale    CUfunction
	fnVecSilu     CUfunction
	fnRmsNorm     CUfunction
)

func initKernels() {
	kernelsOnce.Do(func() {
		if !SgemmReady() {
			return
		}
		// Pre-warm allocator before loading more PTX modules
		var warmPtr CUdeviceptr
		if r := cuMemAlloc(&warmPtr, 512*1024*1024); r == CUDA_SUCCESS {
			cuMemFree(warmPtr)
		}
		var err error
		fnVecAdd, err = LoadPTX(VecAddPTX, "vec_add")
		if err != nil {
			fmt.Printf("[gpu] vec_add failed: %v\n", err)
			return
		}
		fnVecMul, _ = LoadPTX(VecMulPTX, "vec_mul")
		fnVecScale, _ = LoadPTX(VecScalePTX, "vec_scale")
		fnVecSilu, _ = LoadPTX(VecSiLUPTX, "vec_silu")
		fnRmsNorm, _ = LoadPTX(RmsNormPTX, "rms_norm")
		kernelsLoaded = true
		fmt.Println("[gpu] Element-wise kernels loaded (add, mul, scale, silu, rmsnorm)")
	})
}

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
	if b.gpu != nil { return nil }
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
	if kernelsLoaded && n >= 2048 && tryGPU(a, out) {
		a.ToGPU(); b.ToGPU(); out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnVecAdd, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&b.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr), unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	// CPU fallback
	a.ToCPU(); b.ToCPU(); out.ToCPU()
	for i := 0; i < n; i++ {
		out.cpu[i] = a.cpu[i] + b.cpu[i]
	}
}

// Mul: out = a * b (element-wise)
func DevMul(out, a, b *DevBuf) {
	initKernels()
	n := a.n
	if kernelsLoaded && n >= 2048 && tryGPU(a, out) {
		a.ToGPU(); b.ToGPU(); out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnVecMul, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&b.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr), unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	a.ToCPU(); b.ToCPU(); out.ToCPU()
	for i := 0; i < n; i++ {
		out.cpu[i] = a.cpu[i] * b.cpu[i]
	}
}

// Scale: out = a * scalar
func DevScale(out, a *DevBuf, s float32) {
	initKernels()
	n := a.n
	if kernelsLoaded && n >= 2048 && tryGPU(a, out) {
		a.ToGPU(); out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnVecScale, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&s), unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	a.ToCPU(); out.ToCPU()
	for i := 0; i < n; i++ {
		out.cpu[i] = a.cpu[i] * s
	}
}

// SiLU: out = a * sigmoid(a)
func DevSiLU(out, a *DevBuf) {
	initKernels()
	n := a.n
	if kernelsLoaded && n >= 2048 && tryGPU(a, out) {
		a.ToGPU(); out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnVecSilu, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	a.ToCPU(); out.ToCPU()
	for i := 0; i < n; i++ {
		x := a.cpu[i]
		out.cpu[i] = x / (1.0 + float32(math.Exp(float64(-x))))
	}
}

// RMSNorm: out = x * weight * rsqrt(mean(x^2) + eps)
func DevRMSNorm(out, x, weight *DevBuf, eps float32) {
	initKernels()
	n := x.n
	if kernelsLoaded && n >= 2048 && n <= 256*8192 {
		x.ToGPU(); weight.ToGPU(); out.ToGPU()
		nn := uint32(n)
		LaunchKernel(fnRmsNorm, 1, 1, 1, 256, 1, 1, 256*4,
			unsafe.Pointer(&x.gpu.Ptr), unsafe.Pointer(&weight.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&eps))
		out.dev = GPU_DEVICE
		return
	}
	// CPU fallback
	x.ToCPU(); weight.ToCPU(); out.ToCPU()
	var ss float32
	for _, v := range x.cpu[:n] {
		ss += v * v
	}
	ss = 1.0 / float32(math.Sqrt(float64(ss/float32(n)+eps)))
	for i := 0; i < n; i++ {
		out.cpu[i] = x.cpu[i] * ss * weight.cpu[i]
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
	x.ToCPU(); W.ToCPU(); out.ToCPU()
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
		// GPU-to-GPU copy via cuMemcpyDtoD
		src.ToGPU()
		dst.ToGPU()
		cuMemcpyDtoD(dst.gpu.Ptr, src.gpu.Ptr, uint64(src.n*4))
		Sync()
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
	x.ToCPU(); W.ToCPU(); out.ToCPU()
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
	ropeOnce   sync.Once
	ropeFn     CUfunction
	attnFn     CUfunction
	ropeReady  bool
	attnReady  bool
)

func initRoPEAttn() {
	ropeOnce.Do(func() {
		if !SgemmReady() { return }
		var warmPtr CUdeviceptr
		if r := cuMemAlloc(&warmPtr, 256*1024*1024); r == CUDA_SUCCESS { cuMemFree(warmPtr) }
		var err error
		ropeFn, err = LoadPTX(RoPEPTX, "rope_apply")
		if err != nil { fmt.Printf("[gpu] RoPE kernel failed: %v\n", err); return }
		ropeReady = true
		attnFn, err = LoadPTX(AttentionPTX, "gqa_attention")
		if err != nil { fmt.Printf("[gpu] Attention kernel failed: %v\n", err); return }
		attnReady = true
		fmt.Println("[gpu] RoPE + Attention kernels loaded")
	})
}

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

// DevAttention runs GQA attention on GPU.
// q[nHeads*headDim], kCache/vCache[seqLen*kvDim], out[nHeads*headDim]
func DevAttention(out, q, kCache, vCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int) {
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
			unsafe.Pointer(&hd))
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

// InitAllKernels pre-loads all GPU kernels. Call from the CUDA-owning thread.
func InitAllKernels() {
	initKernels()
	initRoPEAttn()
	initQ4()
}

// Fused SiLU*Mul
var (
	fusedSiLUMulOnce sync.Once
	fnFusedSiLUMul   CUfunction
	fusedSiLUMulOK   bool
)

// DevSiLUMul computes out = silu(a) * b in one kernel launch
func DevSiLUMul(out, a, b *DevBuf) {
	fusedSiLUMulOnce.Do(func() {
		if !SgemmReady() { return }
		var err error
		fnFusedSiLUMul, err = LoadPTX(FusedSiLUMulPTX, "fused_silu_mul")
		if err != nil { return }
		fusedSiLUMulOK = true
	})
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
	a.ToCPU(); b.ToCPU(); out.ToCPU()
	for i := 0; i < n; i++ {
		x := a.cpu[i]
		out.cpu[i] = x / (1.0 + float32(math.Exp(float64(-x)))) * b.cpu[i]
	}
}
