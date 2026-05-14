package gpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/runtime/quant"
)

var fnNVFP4DequantF32 CUfunction

// NVFP4KernelKind identifies the packed NVFP4 kernel family a caller wants.
// The interface is intentionally defined before native dispatch exists so the
// fallback and future tensor-core paths agree on dimensions and buffer layout.
type NVFP4KernelKind int

const (
	NVFP4KernelGEMV NVFP4KernelKind = iota
	NVFP4KernelGEMM
)

// NVFP4KernelSpec describes a packed NVFP4 multiply request.
//
// Layout contract:
//   - W is row-major [OutDim, InDim] packed as U8 [OutDim, InDim/2]
//   - WeightScale is row-major F8_E4M3 [OutDim, InDim/GroupSize]
//   - X is F32 row-major [Batch, InDim] for GEMM, or [1, InDim] for GEMV
//   - Out is F32 row-major [Batch, OutDim]
//   - GroupSize is currently required to be 16 for ModelOpt NVFP4 native paths
//
// Native Blackwell/tensor-core dispatch can use this spec without changing the
// public fallback entry points.
type NVFP4KernelSpec struct {
	Kind      NVFP4KernelKind
	OutDim    int
	InDim     int
	Batch     int
	Groups    int
	GroupSize int
}

// NativeNVFP4TensorCoreSupported reports whether the active CUDA device is new
// enough for native NVFP4 tensor-core work. Blackwell-class GPUs are expected to
// expose compute capability 10.x or newer. This is a capability gate only; the
// native kernel path remains disabled until implemented and validated.
func NativeNVFP4TensorCoreSupported() bool {
	if !Available() {
		return false
	}
	major, minor := ComputeCapability()
	return supportsNativeNVFP4TensorCore(major, minor)
}

func supportsNativeNVFP4TensorCore(major, minor int) bool {
	return major >= 10
}

// ValidateNVFP4KernelSpec checks packed NVFP4 GEMV/GEMM dimensions without
// requiring GPU buffers. It is the shared shape gate for future native kernels.
func ValidateNVFP4KernelSpec(spec NVFP4KernelSpec) error {
	if spec.Kind != NVFP4KernelGEMV && spec.Kind != NVFP4KernelGEMM {
		return fmt.Errorf("invalid NVFP4 kernel kind %d", spec.Kind)
	}
	if spec.OutDim <= 0 || spec.InDim <= 0 || spec.Batch <= 0 || spec.Groups <= 0 || spec.GroupSize <= 0 {
		return fmt.Errorf("invalid NVFP4 kernel dims out=%d in=%d batch=%d groups=%d groupSize=%d", spec.OutDim, spec.InDim, spec.Batch, spec.Groups, spec.GroupSize)
	}
	if spec.OutDim > math.MaxUint32 || spec.InDim > math.MaxUint32 || spec.Batch > math.MaxUint32 || spec.Groups > math.MaxUint32 || spec.GroupSize > math.MaxUint32 {
		return fmt.Errorf("NVFP4 kernel dims exceed CUDA u32 interface out=%d in=%d batch=%d groups=%d groupSize=%d", spec.OutDim, spec.InDim, spec.Batch, spec.Groups, spec.GroupSize)
	}
	if spec.Kind == NVFP4KernelGEMV && spec.Batch != 1 {
		return fmt.Errorf("NVFP4 GEMV batch=%d, want 1", spec.Batch)
	}
	if spec.InDim%2 != 0 || spec.InDim%spec.GroupSize != 0 || spec.Groups != spec.InDim/spec.GroupSize {
		return fmt.Errorf("NVFP4 kernel group layout mismatch in=%d groups=%d groupSize=%d", spec.InDim, spec.Groups, spec.GroupSize)
	}
	if spec.GroupSize != 16 {
		return fmt.Errorf("NVFP4 native kernels require groupSize=16, got %d", spec.GroupSize)
	}
	if _, ok := checkedMulInt(spec.OutDim, spec.InDim/2); !ok {
		return fmt.Errorf("NVFP4 packed weight bytes overflow out=%d in=%d", spec.OutDim, spec.InDim)
	}
	if _, ok := checkedMulInt(spec.OutDim, spec.Groups); !ok {
		return fmt.Errorf("NVFP4 scale bytes overflow out=%d groups=%d", spec.OutDim, spec.Groups)
	}
	if _, ok := checkedMulInt(spec.Batch, spec.InDim); !ok {
		return fmt.Errorf("NVFP4 input elements overflow batch=%d in=%d", spec.Batch, spec.InDim)
	}
	if _, ok := checkedMulInt(spec.Batch, spec.OutDim); !ok {
		return fmt.Errorf("NVFP4 output elements overflow batch=%d out=%d", spec.Batch, spec.OutDim)
	}
	return nil
}

// GPUNVFP4Weight is the GPU-resident representation for ModelOpt/NVFP4
// weights. It is deliberately separate from GPTQ/MLX upload structures because
// NVFP4 uses U8 packed FP4 weights plus F8_E4M3 per-block scales and scalar
// scale metadata, not affine INT4 or GPTQ group-index metadata.
type GPUNVFP4Weight struct {
	Weight        *Buffer // raw U8 packed FP4 bytes, padded to float32 allocation granularity
	WeightScale   *Buffer // raw F8_E4M3 scale bytes, padded to float32 allocation granularity
	WeightScale2  float32
	InputScale    float32
	HasInputScale bool
	OutDim        int
	InDim         int
	Groups        int
	GroupSize     int
	WeightBytes   int
	ScaleBytes    int
}

// UploadNVFP4Weight uploads the observed NVFP4 packed representation to GPU
// memory without converting it to MLX/GPTQ. Kernel dispatch is added separately.
func UploadNVFP4Weight(qw *quant.NVFP4Weight) (*GPUNVFP4Weight, error) {
	if err := quant.ValidateNVFP4Weight(qw); err != nil {
		return nil, err
	}
	if !SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}
	EnsureContext()

	weightBytes, scaleBytes, err := nvfp4RequiredBytes(qw.OutDim, qw.InDim, qw.Groups)
	if err != nil {
		return nil, err
	}
	w := &GPUNVFP4Weight{
		WeightScale2:  qw.WeightScale2,
		InputScale:    qw.InputScale,
		HasInputScale: qw.HasInputScale,
		OutDim:        qw.OutDim,
		InDim:         qw.InDim,
		Groups:        qw.Groups,
		GroupSize:     qw.GroupSize,
		WeightBytes:   weightBytes,
		ScaleBytes:    scaleBytes,
	}

	weightUpload := bytesAsFloat32Padded(qw.Weight[:weightBytes])
	wb, err := Malloc(len(weightUpload))
	if err != nil {
		return nil, fmt.Errorf("alloc NVFP4 weight (%d bytes): %w", weightBytes, err)
	}
	w.Weight = wb
	if err := wb.Upload(weightUpload); err != nil {
		w.Free()
		return nil, fmt.Errorf("upload NVFP4 weight: %w", err)
	}

	scaleUpload := bytesAsFloat32Padded(qw.WeightScale[:scaleBytes])
	sb, err := Malloc(len(scaleUpload))
	if err != nil {
		w.Free()
		return nil, fmt.Errorf("alloc NVFP4 weight_scale (%d bytes): %w", scaleBytes, err)
	}
	w.WeightScale = sb
	if err := sb.Upload(scaleUpload); err != nil {
		w.Free()
		return nil, fmt.Errorf("upload NVFP4 weight_scale: %w", err)
	}
	return w, nil
}

func (w *GPUNVFP4Weight) Free() {
	if w == nil {
		return
	}
	if w.Weight != nil {
		w.Weight.Free()
		w.Weight = nil
	}
	if w.WeightScale != nil {
		w.WeightScale.Free()
		w.WeightScale = nil
	}
}

// DequantNVFP4ToF32 materializes a GPU-resident NVFP4 weight as row-major F32.
// It first tries the CUDA dequant kernel, then falls back to downloading raw
// buffers and reusing the runtime/quant reference dequantizer.
func DequantNVFP4ToF32(w *GPUNVFP4Weight) ([]float32, error) {
	if !validGPUNVFP4Weight(w) {
		return nil, fmt.Errorf("invalid GPU NVFP4 weight")
	}
	if out, ok := dequantNVFP4ToF32CUDA(w); ok {
		return out, nil
	}
	weightPacked := make([]float32, f32SlotsForBytes(w.WeightBytes))
	if err := w.Weight.Download(weightPacked); err != nil {
		return nil, fmt.Errorf("download NVFP4 weight: %w", err)
	}
	scalePacked := make([]float32, f32SlotsForBytes(w.ScaleBytes))
	if err := w.WeightScale.Download(scalePacked); err != nil {
		return nil, fmt.Errorf("download NVFP4 weight_scale: %w", err)
	}
	qw := &quant.NVFP4Weight{
		Weight:        float32PackedAsBytes(weightPacked, w.WeightBytes),
		WeightScale:   float32PackedAsBytes(scalePacked, w.ScaleBytes),
		WeightScale2:  w.WeightScale2,
		InputScale:    w.InputScale,
		HasInputScale: w.HasInputScale,
		OutDim:        w.OutDim,
		InDim:         w.InDim,
		Groups:        w.Groups,
		GroupSize:     w.GroupSize,
	}
	out := quant.DequantNVFP4(qw)
	if out == nil {
		return nil, fmt.Errorf("dequantize NVFP4 fallback failed")
	}
	return out, nil
}

// GemvNVFP4 computes dense out[outDim] = W_nvfp4[outDim,inDim] · x[inDim].
// WARNING: this fallback materializes OutDim*InDim F32 weights per call; native
// packed GEMV/GEMM kernels can replace this behind the same entry point.
func GemvNVFP4(out, x []float32, w *GPUNVFP4Weight) error {
	if !validGPUNVFP4Weight(w) {
		return fmt.Errorf("invalid GPU NVFP4 weight")
	}
	if len(out) < w.OutDim || len(x) < w.InDim {
		return fmt.Errorf("invalid NVFP4 GEMV buffers out=%d/%d x=%d/%d", len(out), w.OutDim, len(x), w.InDim)
	}
	weights, err := DequantNVFP4ToF32(w)
	if err != nil {
		return err
	}
	return gemvNVFP4F32(out, x, w.OutDim, w.InDim, weights)
}

func gemvNVFP4F32(out, x []float32, outDim, inDim int, weights []float32) error {
	wantWeights, ok := checkedMulInt(outDim, inDim)
	if outDim <= 0 || inDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(weights) < wantWeights {
		return fmt.Errorf("invalid NVFP4 F32 GEMV buffers out=%d/%d x=%d/%d weights=%d/%d", len(out), outDim, len(x), inDim, len(weights), wantWeights)
	}
	for row := 0; row < outDim; row++ {
		sum := float32(0)
		rowOff := row * inDim
		for col := 0; col < inDim; col++ {
			sum += weights[rowOff+col] * x[col]
		}
		out[row] = sum
	}
	return nil
}

func dequantNVFP4ToF32CUDA(w *GPUNVFP4Weight) ([]float32, bool) {
	if fnNVFP4DequantF32 == 0 || !megaModuleOK {
		return nil, false
	}
	outLen, ok := checkedMulInt(w.OutDim, w.InDim)
	if !ok || outLen > math.MaxUint32 {
		return nil, false
	}
	outBuf, err := Malloc(outLen)
	if err != nil {
		debugf("[gpu] NVFP4 CUDA dequant alloc fallback: %v\n", err)
		return nil, false
	}
	defer outBuf.Free()

	total := uint32(outLen)
	inDim := uint32(w.InDim)
	groupSize := uint32(w.GroupSize)
	grid := (total + 255) / 256
	if err := LaunchKernel(fnNVFP4DequantF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&w.Weight.Ptr),
		unsafe.Pointer(&w.WeightScale.Ptr),
		unsafe.Pointer(&outBuf.Ptr),
		unsafe.Pointer(&w.WeightScale2),
		unsafe.Pointer(&total),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&groupSize)); err != nil {
		debugf("[gpu] NVFP4 CUDA dequant launch fallback: %v\n", err)
		return nil, false
	}
	if err := SyncErr(); err != nil {
		debugf("[gpu] NVFP4 CUDA dequant sync fallback: %v\n", err)
		return nil, false
	}
	out := make([]float32, outLen)
	if err := outBuf.Download(out); err != nil {
		debugf("[gpu] NVFP4 CUDA dequant download fallback: %v\n", err)
		return nil, false
	}
	return out, true
}

func validGPUNVFP4Weight(w *GPUNVFP4Weight) bool {
	if w == nil || w.Weight == nil || w.WeightScale == nil || w.OutDim <= 0 || w.InDim <= 0 || w.Groups <= 0 || w.GroupSize <= 0 {
		return false
	}
	if w.InDim%2 != 0 || w.InDim%w.GroupSize != 0 || w.Groups != w.InDim/w.GroupSize {
		return false
	}
	weightBytes, scaleBytes, err := nvfp4RequiredBytes(w.OutDim, w.InDim, w.Groups)
	if err != nil || w.WeightBytes != weightBytes || w.ScaleBytes != scaleBytes {
		return false
	}
	return hasPaddedByteCapacity(w.Weight.Size, weightBytes) && hasPaddedByteCapacity(w.WeightScale.Size, scaleBytes)
}

func nvfp4RequiredBytes(outDim, inDim, groups int) (int, int, error) {
	if outDim <= 0 || inDim <= 0 || groups <= 0 || inDim%2 != 0 {
		return 0, 0, fmt.Errorf("invalid NVFP4 byte dims out=%d in=%d groups=%d", outDim, inDim, groups)
	}
	weightBytes, ok := checkedMulInt(outDim, inDim/2)
	if !ok {
		return 0, 0, fmt.Errorf("NVFP4 weight byte size overflows out=%d in=%d", outDim, inDim)
	}
	scaleBytes, ok := checkedMulInt(outDim, groups)
	if !ok {
		return 0, 0, fmt.Errorf("NVFP4 scale byte size overflows out=%d groups=%d", outDim, groups)
	}
	return weightBytes, scaleBytes, nil
}

func f32SlotsForBytes(n int) int {
	if n <= 0 {
		return 0
	}
	return n/4 + boolToInt(n%4 != 0)
}

func hasPaddedByteCapacity(sizeBytes, requiredBytes int) bool {
	if requiredBytes <= 0 {
		return sizeBytes >= 0
	}
	maxInt := int(^uint(0) >> 1)
	slots := f32SlotsForBytes(requiredBytes)
	if slots > maxInt/4 {
		return false
	}
	return sizeBytes >= slots*4
}

func bytesAsFloat32Padded(data []byte) []float32 {
	out := make([]float32, f32SlotsForBytes(len(data)))
	for i := range out {
		off := i * 4
		if off+4 <= len(data) {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4]))
		} else {
			var tmp [4]byte
			copy(tmp[:], data[off:])
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(tmp[:]))
		}
	}
	return out
}

func float32PackedAsBytes(data []float32, n int) []byte {
	if n <= 0 || len(data) == 0 {
		return nil
	}
	maxBytes, ok := checkedMulInt(len(data), 4)
	if !ok || n > maxBytes {
		return nil
	}
	out := make([]byte, maxBytes)
	for i, f := range data {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(f))
	}
	return out[:n]
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
