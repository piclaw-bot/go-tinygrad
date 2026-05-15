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
	fnVecAddScaled   CUfunction
	fnToBF16F32      CUfunction
	fnVecSilu        CUfunction
	fnRmsNorm        CUfunction
	fnRmsNormNoScale CUfunction
	fnGELUTanhMul    CUfunction
)

func initKernels() { loadMegaModule() }

// NewDevBuf creates a CPU buffer.
func NewDevBuf(n int) *DevBuf {
	if n < 0 {
		n = 0
	}
	return &DevBuf{cpu: make([]float32, n), n: n, dev: CPU}
}

// NewDevBufFrom wraps existing CPU data.
func NewDevBufFrom(data []float32) *DevBuf {
	return &DevBuf{cpu: data, n: len(data), dev: CPU}
}

// NewDevBufGPU allocates a GPU-only buffer without uploading zeroed CPU data.
// Its contents are undefined until overwritten by a GPU operation.
func NewDevBufGPU(n int) (*DevBuf, error) {
	if n < 0 {
		n = 0
	}
	if !SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}
	buf, err := Malloc(n)
	if err != nil {
		return nil, err
	}
	return &DevBuf{gpu: buf, n: n, dev: GPU_DEVICE, ownGPU: true}, nil
}

// ToGPU ensures data is on GPU. No-op if already there.
func (b *DevBuf) ToGPU() error {
	if b == nil {
		return fmt.Errorf("nil DevBuf")
	}
	if b.gpu != nil {
		if b.dev == CPU && b.cpu != nil {
			// CPU data was modified — re-upload
			if err := b.gpu.Upload(b.cpu); err != nil {
				return err
			}
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
		if err := b.gpu.Upload(b.cpu); err != nil {
			if b.ownGPU && b.gpu != nil {
				b.gpu.Free()
			}
			b.gpu = nil
			b.ownGPU = false
			return err
		}
	}
	b.dev = GPU_DEVICE
	return nil
}

// ToCPU ensures data is on CPU. No-op if already there.
func (b *DevBuf) ToCPU() {
	if b == nil {
		return
	}
	if b.cpu == nil {
		b.cpu = make([]float32, b.n)
	}
	if b.gpu != nil && b.dev == GPU_DEVICE {
		if err := b.gpu.Download(b.cpu); err != nil {
			// Keep GPU as authoritative if the download fails; callers may still inspect
			// the CPU backing slice, but a later GPU op must not treat stale CPU data as
			// newer and re-upload it over valid device contents.
			return
		}
	}
	b.dev = CPU
}

// Data returns CPU-side data (downloading from GPU if needed).
func (b *DevBuf) Data() []float32 {
	if b == nil {
		return nil
	}
	b.ToCPU()
	return b.cpu
}

// EnsureGPU ensures GPU buffer exists without uploading CPU data.
func (b *DevBuf) EnsureGPU() error {
	if b == nil {
		return fmt.Errorf("nil DevBuf")
	}
	if b.gpu != nil {
		return nil
	}
	return b.ToGPU()
}

// GPUPtr returns the GPU buffer, uploading if needed.
func (b *DevBuf) GPUPtr() *Buffer {
	if b == nil {
		return nil
	}
	if b.gpu == nil {
		if err := b.ToGPU(); err != nil {
			return nil
		}
	}
	return b.gpu
}

// Len returns element count.
func (b *DevBuf) Len() int {
	if b == nil {
		return 0
	}
	return b.n
}

// OnGPU returns true if data is authoritatively on GPU.
func (b *DevBuf) OnGPU() bool { return b != nil && b.dev == GPU_DEVICE && b.gpu != nil }

// tryGPU attempts to move buffers to GPU. Returns true if all succeeded.
func tryGPU(bufs ...*DevBuf) bool {
	for _, b := range bufs {
		if b == nil || b.ToGPU() != nil || b.gpu == nil {
			return false
		}
	}
	return true
}

func commonLen(bufs ...*DevBuf) int {
	if len(bufs) == 0 {
		return 0
	}
	n := -1
	for _, b := range bufs {
		if b == nil {
			return 0
		}
		if n < 0 || b.n < n {
			n = b.n
		}
	}
	if n < 0 {
		return 0
	}
	return n
}

// --- Ops: dispatch to GPU if possible, CPU fallback ---

// Add: out = a + b (element-wise)
func DevAdd(out, a, b *DevBuf) {
	initKernels()
	n := commonLen(a, b, out)
	if n <= 0 {
		return
	}
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
	n := commonLen(a, b, out)
	if n <= 0 {
		return
	}
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
	n := commonLen(a, out)
	if n <= 0 {
		return
	}
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

// AddScaled: out = a + b * scalar.
func DevAddScaled(out, a, b *DevBuf, s float32) {
	initKernels()
	n := commonLen(a, b, out)
	if n <= 0 {
		return
	}
	if kernelsLoaded && fnVecAddScaled != 0 && tryGPU(a, b, out) {
		nn := uint32(n)
		LaunchKernel(fnVecAddScaled, (uint32(n)+255)/256, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&a.gpu.Ptr), unsafe.Pointer(&b.gpu.Ptr), unsafe.Pointer(&out.gpu.Ptr), unsafe.Pointer(&s), unsafe.Pointer(&nn))
		out.dev = GPU_DEVICE
		return
	}
	a.ToCPU()
	b.ToCPU()
	out.ToCPU()
	for i := 0; i < n; i++ {
		out.cpu[i] = a.cpu[i] + b.cpu[i]*s
	}
}

// ToBF16: truncate float32 values in-place to BF16 precision.
func DevToBF16(x *DevBuf, n int) {
	initKernels()
	if x == nil {
		return
	}
	if n <= 0 || n > x.n {
		n = x.n
	}
	if n <= 0 {
		return
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
	for i := 0; i < n; i++ {
		bits := math.Float32bits(x.cpu[i])
		bits &= 0xFFFF0000
		x.cpu[i] = math.Float32frombits(bits)
	}
}

// SiLU: out = a * sigmoid(a)
func DevSiLU(out, a *DevBuf) {
	initKernels()
	n := commonLen(a, out)
	if n <= 0 {
		return
	}
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
	n := commonLen(x, weight, out) // weight size remains the practical canonical dimension, bounded by x/out.
	if n <= 0 {
		return
	}
	if kernelsLoaded && n <= 256*8192 && tryGPU(x, weight, out) {
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
	n := commonLen(x, out)
	if n <= 0 {
		return
	}
	if kernelsLoaded && fnRmsNormNoScale != 0 && n <= 256*8192 && tryGPU(x, out) {
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
	if out == nil || x == nil || W == nil || M <= 0 || K <= 0 || out.n < M || x.n < K || W.n < M*K {
		return
	}
	if SgemmReady() && tryGPU(x, W, out) {
		Sgemm(M, 1, K, 1.0, W.gpu, x.gpu, out.gpu)
		out.dev = GPU_DEVICE
		return
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
	if x == nil {
		return
	}
	if n <= 0 || n > x.n {
		n = x.n
	}
	if n <= 0 {
		return
	}
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
	if dst == nil || src == nil {
		return
	}
	n := commonLen(dst, src)
	if n <= 0 {
		return
	}
	if src.gpu != nil && dst.gpu != nil && n >= 2048 {
		if src.ToGPU() == nil && dst.ToGPU() == nil {
			recordDeviceToDeviceCopy()
			cuMemcpyDtoDAsync(dst.gpu.Ptr, src.gpu.Ptr, uint64(n)*4, 0) // stream 0 = default
			dst.dev = GPU_DEVICE
			return
		}
	}
	src.ToCPU()
	dst.ToCPU()
	copy(dst.cpu[:n], src.cpu[:n])
}

// MarkDirty marks CPU data as authoritative (will re-upload on next GPU access).
func (b *DevBuf) MarkDirty() {
	if b != nil {
		b.dev = CPU
	}
}

// MarkOnGPU marks GPU data as authoritative after in-place GPU-side mutation.
func (b *DevBuf) MarkOnGPU() {
	if b != nil {
		b.dev = GPU_DEVICE
	}
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
	if out == nil || x == nil || W == nil || K <= 0 || N <= 0 || out.n < N || x.n < K || W.n < K*N {
		return
	}
	if SgemmReady() && tryGPU(x, W, out) {
		Sgemm(1, N, K, 1.0, x.gpu, W.gpu, out.gpu)
		out.dev = GPU_DEVICE
		return
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

// CopyDtoD wraps cuMemcpyDtoD for direct GPU→GPU copy.
func CopyDtoD(dst, src CUdeviceptr, bytes uint64) error {
	if dst == 0 || src == 0 || bytes == 0 {
		return nil
	}
	EnsureContext()
	if r := cuMemcpyDtoD(dst, src, bytes); r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyDtoD: error %d", r)
	}
	recordDeviceToDeviceCopy()
	return nil
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
	n := commonLen(a, b, out)
	if n <= 0 {
		return
	}
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
	maxN := commonLen(gate, up)
	if n <= 0 || n > maxN {
		n = maxN
	}
	if n <= 0 {
		return
	}
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
	if b == nil || offset < 0 || n < 0 || offset > b.n {
		return NewDevBuf(0)
	}
	if n > b.n-offset {
		n = b.n - offset
	}
	s := &DevBuf{n: n, dev: b.dev, ownGPU: false}
	if b.cpu != nil && offset+n <= len(b.cpu) {
		s.cpu = b.cpu[offset : offset+n]
	}
	if b.gpu != nil {
		offsetBytes, errOffset := checkedByteSize(offset, 0)
		sizeBytes, errSize := checkedByteSize(n, 0)
		if errOffset == nil && errSize == nil {
			s.gpu = &Buffer{
				Ptr:  b.gpu.Ptr + CUdeviceptr(offsetBytes),
				Size: int(sizeBytes),
			}
		}
	}
	return s
}
