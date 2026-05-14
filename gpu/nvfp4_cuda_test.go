package gpu

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/quant"
)

func TestSupportsNativeNVFP4TensorCore(t *testing.T) {
	cases := []struct {
		major int
		minor int
		want  bool
	}{
		{8, 0, false},
		{8, 9, false},
		{9, 0, false},
		{10, 0, true},
		{12, 0, true},
	}
	for _, tc := range cases {
		if got := supportsNativeNVFP4TensorCore(tc.major, tc.minor); got != tc.want {
			t.Fatalf("supportsNativeNVFP4TensorCore(%d,%d)=%v want %v", tc.major, tc.minor, got, tc.want)
		}
	}
}

func TestValidateNVFP4KernelSpec(t *testing.T) {
	good := NVFP4KernelSpec{Kind: NVFP4KernelGEMM, OutDim: 4096, InDim: 4096, Batch: 4, Groups: 256, GroupSize: 16}
	if err := ValidateNVFP4KernelSpec(good); err != nil {
		t.Fatalf("ValidateNVFP4KernelSpec: %v", err)
	}
	bad := []NVFP4KernelSpec{
		{Kind: NVFP4KernelGEMV, OutDim: 4096, InDim: 4096, Batch: 2, Groups: 256, GroupSize: 16},
		{Kind: NVFP4KernelGEMM, OutDim: 4096, InDim: 4095, Batch: 1, Groups: 256, GroupSize: 16},
		{Kind: NVFP4KernelGEMM, OutDim: 4096, InDim: 4096, Batch: 1, Groups: 128, GroupSize: 16},
		{Kind: NVFP4KernelGEMM, OutDim: 4096, InDim: 4096, Batch: 1, Groups: 64, GroupSize: 64},
		{Kind: NVFP4KernelGEMM, OutDim: int(math.MaxUint32) + 1, InDim: 16, Batch: 1, Groups: 1, GroupSize: 16},
	}
	for _, spec := range bad {
		if err := ValidateNVFP4KernelSpec(spec); err == nil {
			t.Fatalf("ValidateNVFP4KernelSpec accepted %+v", spec)
		}
	}
}

func TestNVFP4RequiredBytes(t *testing.T) {
	weightBytes, scaleBytes, err := nvfp4RequiredBytes(4096, 4096, 256)
	if err != nil {
		t.Fatalf("nvfp4RequiredBytes: %v", err)
	}
	if weightBytes != 4096*2048 || scaleBytes != 4096*256 {
		t.Fatalf("weightBytes=%d scaleBytes=%d", weightBytes, scaleBytes)
	}
}

func TestNVFP4RequiredBytesRejectsBadDims(t *testing.T) {
	if _, _, err := nvfp4RequiredBytes(1, 3, 1); err == nil {
		t.Fatal("nvfp4RequiredBytes accepted odd inDim")
	}
}

func TestF32SlotsForBytesAvoidsOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	got := f32SlotsForBytes(maxInt)
	want := maxInt/4 + 1
	if got != want {
		t.Fatalf("f32SlotsForBytes(maxInt)=%d want %d", got, want)
	}
}

func TestHasPaddedByteCapacityAvoidsOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if hasPaddedByteCapacity(maxInt, maxInt) {
		t.Fatal("hasPaddedByteCapacity accepted unrepresentable padded byte count")
	}
	if !hasPaddedByteCapacity(8, 5) {
		t.Fatal("hasPaddedByteCapacity rejected valid padded byte count")
	}
}

func TestBytesAsFloat32PaddedRoundTripsRawBytes(t *testing.T) {
	input := []byte{0x01, 0x02, 0x03, 0x04, 0x05}
	got := bytesAsFloat32Padded(input)
	if len(got) != 2 {
		t.Fatalf("len=%d want 2", len(got))
	}
	if math.Float32bits(got[0]) != 0x04030201 {
		t.Fatalf("bits[0]=%#x", math.Float32bits(got[0]))
	}
	if math.Float32bits(got[1]) != 0x00000005 {
		t.Fatalf("bits[1]=%#x", math.Float32bits(got[1]))
	}
	if roundTrip := float32PackedAsBytes(got, len(input)); string(roundTrip) != string(input) {
		t.Fatalf("roundTrip=%#v want %#v", roundTrip, input)
	}
	if roundTrip := float32PackedAsBytes(got, len(got)*4+1); roundTrip != nil {
		t.Fatalf("oversized roundTrip=%#v want nil", roundTrip)
	}
}

func TestGemvNVFP4RejectsInvalidBuffers(t *testing.T) {
	w := &GPUNVFP4Weight{
		Weight:      &Buffer{Size: f32SlotsForBytes(8) * 4},
		WeightScale: &Buffer{Size: f32SlotsForBytes(2) * 4},
		OutDim:      2,
		InDim:       8,
		Groups:      1,
		GroupSize:   8,
		WeightBytes: 8,
		ScaleBytes:  2,
	}
	if err := GemvNVFP4(make([]float32, 1), make([]float32, 8), w); err == nil {
		t.Fatal("GemvNVFP4 accepted short output")
	}
	if err := GemvNVFP4(make([]float32, 2), make([]float32, 7), w); err == nil {
		t.Fatal("GemvNVFP4 accepted short input")
	}
}

func TestGemvNVFP4F32RejectsOverflowShape(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if err := gemvNVFP4F32(make([]float32, 1), make([]float32, 1), maxInt/2+1, 3, make([]float32, 1)); err == nil {
		t.Fatal("gemvNVFP4F32 accepted overflowing shape")
	}
}

func TestGemvNVFP4F32WithReferenceDequant(t *testing.T) {
	qw := syntheticNVFP4Weight()
	weights := quant.DequantNVFP4(qw)
	x := []float32{1, -1, 2, -2, 0.5, -0.5, 3, -3, 4, -4, 1.5, -1.5, 2.5, -2.5, 0.25, -0.25}
	want := make([]float32, 2)
	for row := 0; row < qw.OutDim; row++ {
		for col := 0; col < qw.InDim; col++ {
			want[row] += weights[row*qw.InDim+col] * x[col]
		}
	}
	got := make([]float32, 2)
	if err := gemvNVFP4F32(got, x, qw.OutDim, qw.InDim, weights); err != nil {
		t.Fatalf("gemvNVFP4F32: %v", err)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestDequantNVFP4ToF32CUDAMatchesCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("no GPU")
	}
	qw := syntheticNVFP4Weight()
	gw, err := UploadNVFP4Weight(qw)
	if err != nil {
		t.Fatalf("UploadNVFP4Weight: %v", err)
	}
	defer gw.Free()

	got, ok := dequantNVFP4ToF32CUDA(gw)
	if !ok {
		t.Skip("NVFP4 CUDA dequant kernel unavailable")
	}
	want := quant.DequantNVFP4(qw)
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func syntheticNVFP4Weight() *quant.NVFP4Weight {
	return &quant.NVFP4Weight{
		Weight: []byte{
			0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
			0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
		},
		WeightScale:  []byte{0x38, 0x40}, // 1.0, 2.0
		WeightScale2: 0.5,
		OutDim:       2,
		InDim:        16,
		Groups:       1,
		GroupSize:    16,
	}
}

func TestDequantNVFP4ToF32CUDARejectsUint32Overflow(t *testing.T) {
	oldFn, oldOK := fnNVFP4DequantF32, megaModuleOK
	defer func() { fnNVFP4DequantF32, megaModuleOK = oldFn, oldOK }()
	fnNVFP4DequantF32 = 1
	megaModuleOK = true
	maxUint32 := int(^uint32(0))
	w := &GPUNVFP4Weight{OutDim: maxUint32/2 + 1, InDim: 3}
	if out, ok := dequantNVFP4ToF32CUDA(w); ok || out != nil {
		t.Fatalf("dequantNVFP4ToF32CUDA accepted >uint32 element count")
	}
}

func TestValidGPUNVFP4Weight(t *testing.T) {
	w := &GPUNVFP4Weight{
		Weight:      &Buffer{Size: f32SlotsForBytes(8) * 4},
		WeightScale: &Buffer{Size: f32SlotsForBytes(2) * 4},
		OutDim:      2,
		InDim:       8,
		Groups:      1,
		GroupSize:   8,
		WeightBytes: 8,
		ScaleBytes:  2,
	}
	if !validGPUNVFP4Weight(w) {
		t.Fatal("validGPUNVFP4Weight rejected valid synthetic layout")
	}
	w.ScaleBytes = 1
	if validGPUNVFP4Weight(w) {
		t.Fatal("validGPUNVFP4Weight accepted wrong scale byte count")
	}
}
