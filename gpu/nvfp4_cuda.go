package gpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/runtime/quant"
)

var fnNVFP4DequantF32 CUfunction

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
// It is the correctness fallback used until a native/non-native CUDA dequant
// kernel is wired in. The current implementation downloads the raw byte buffers
// and reuses the runtime/quant reference dequantizer, preserving the same API
// shape that the CUDA kernel will fill later.
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
// The initial integration path materializes W through the F32 dequant fallback;
// native packed GEMV/GEMM kernels can replace this behind the same entry point.
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
	if outDim <= 0 || inDim <= 0 || len(out) < outDim || len(x) < inDim || len(weights) < outDim*inDim {
		return fmt.Errorf("invalid NVFP4 F32 GEMV buffers out=%d/%d x=%d/%d weights=%d/%d", len(out), outDim, len(x), inDim, len(weights), outDim*inDim)
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
	if !ok {
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
	return w.Weight.Size >= f32SlotsForBytes(weightBytes)*4 && w.WeightScale.Size >= f32SlotsForBytes(scaleBytes)*4
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
	return (n + 3) / 4
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
	out := make([]byte, len(data)*4)
	for i, f := range data {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(f))
	}
	return out[:n]
}
