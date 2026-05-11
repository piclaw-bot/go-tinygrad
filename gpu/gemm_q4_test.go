package gpu

import (
	"math"
	"math/rand"
	"testing"
)

func TestGemmQ4Batch(t *testing.T) {
	if !SgemmReady() || !Q4Ready() || !BatchGEMMReady() {
		t.Skip("no GPU or Q4/batch kernels")
	}

	// Create a small test: inDim=256, outDim=128, B=4
	inDim := 256
	outDim := 128
	B := 4
	groupSize := 128
	groups := inDim / groupSize

	// Create quantized weight data
	qweight := make([]int32, (inDim/8)*outDim)
	gidx := make([]int32, inDim)
	scales := make([]float32, groups*outDim)

	rng := rand.New(rand.NewSource(42))
	for i := range qweight {
		qweight[i] = rng.Int31()
	}
	for i := 0; i < inDim; i++ {
		gidx[i] = int32(i / groupSize)
	}
	for i := range scales {
		scales[i] = rng.Float32()*0.1 + 0.01
	}

	// Create input [B, inDim]
	input := make([]float32, B*inDim)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	// CPU reference: dequantize and multiply
	cpuOut := make([]float32, B*outDim)
	for b := 0; b < B; b++ {
		for col := 0; col < outDim; col++ {
			sum := float32(0)
			for i := 0; i < inDim; i++ {
				packed := qweight[(i/8)*outDim+col]
				shift := uint((i % 8) * 4)
				val := int32((packed>>shift)&0xF) - 8
				scale := scales[gidx[i]*int32(outDim)+int32(col)]
				w := float32(val) * scale
				sum += input[b*inDim+i] * w
			}
			cpuOut[b*outDim+col] = sum
		}
	}

	// Upload to GPU
	w, err := UploadQuantWeight(qweight, gidx, scales, inDim, outDim)
	if err != nil {
		t.Fatalf("upload: %v", err)
	}

	inp := NewDevBuf(B * inDim)
	copy(inp.Data(), input)
	inp.MarkDirty()

	out := NewDevBuf(B * outDim)

	// Run batched GEMM
	GemmQ4(out, inp, w, B)
	Sync()

	// Compare
	gpuOut := out.Data()
	maxDiff := float32(0)
	for i := 0; i < B*outDim; i++ {
		d := float32(math.Abs(float64(gpuOut[i] - cpuOut[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}

	t.Logf("Batched Q4 GEMM [%d×%d]×[%d×%d]: maxDiff=%e", B, inDim, inDim, outDim, maxDiff)
	if maxDiff > 0.01 {
		t.Fatalf("drift too high: %e (gpu[0]=%v cpu[0]=%v)", maxDiff, gpuOut[0], cpuOut[0])
	}

	// Also test B=1 (should match single GEMV)
	inp1 := NewDevBuf(inDim)
	copy(inp1.Data(), input[:inDim])
	inp1.MarkDirty()

	out1 := NewDevBuf(outDim)
	GemmQ4(out1, inp1, w, 1)
	Sync()

	gpuOut1 := out1.Data()
	maxDiff1 := float32(0)
	for i := 0; i < outDim; i++ {
		d := float32(math.Abs(float64(gpuOut1[i] - cpuOut[i])))
		if d > maxDiff1 {
			maxDiff1 = d
		}
	}
	t.Logf("B=1 vs CPU: maxDiff=%e", maxDiff1)
	if maxDiff1 > 0.01 {
		t.Fatalf("B=1 drift: %e", maxDiff1)
	}
}

func TestGemmQ4VsGemv(t *testing.T) {
	if !SgemmReady() || !Q4Ready() || !BatchGEMMReady() {
		t.Skip("no GPU or Q4/batch kernels")
	}

	// Compare GEMM(B=1) vs GEMV for same input
	inDim := 256
	outDim := 128
	groupSize := 128
	groups := inDim / groupSize

	rng := rand.New(rand.NewSource(99))
	qweight := make([]int32, (inDim/8)*outDim)
	gidx := make([]int32, inDim)
	scales := make([]float32, groups*outDim)
	for i := range qweight {
		qweight[i] = rng.Int31()
	}
	for i := 0; i < inDim; i++ {
		gidx[i] = int32(i / groupSize)
	}
	for i := range scales {
		scales[i] = rng.Float32()*0.1 + 0.01
	}

	w, _ := UploadQuantWeight(qweight, gidx, scales, inDim, outDim)

	input := make([]float32, inDim)
	for i := range input {
		input[i] = rng.Float32()*2 - 1
	}

	inp := NewDevBuf(inDim)
	copy(inp.Data(), input)
	inp.MarkDirty()

	outGemv := NewDevBuf(outDim)
	outGemm := NewDevBuf(outDim)

	GemvQ4(outGemv, inp, w)
	GemmQ4(outGemm, inp, w, 1)
	Sync()

	gv := outGemv.Data()
	gm := outGemm.Data()
	maxDiff := float32(0)
	for i := 0; i < outDim; i++ {
		d := float32(math.Abs(float64(gv[i] - gm[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("GEMV vs GEMM(B=1): maxDiff=%e", maxDiff)
	if maxDiff > 0.001 {
		t.Fatalf("GEMV/GEMM mismatch: %e", maxDiff)
	}
}

func TestUploadQuantWeightValidation(t *testing.T) {
	if _, err := UploadQuantWeight(nil, nil, nil, 0, 8); err == nil {
		t.Fatalf("expected invalid dimension error")
	}
	if !Q4Ready() {
		t.Skip("Q4 kernel not ready; dimension validation checked before readiness")
	}
	if _, err := UploadQuantWeight(make([]int32, 1), make([]int32, 8), make([]float32, 8), 16, 8); err == nil {
		t.Fatalf("expected short qweight error")
	}
	if _, err := UploadQuantWeight(make([]int32, 16), make([]int32, 16), make([]float32, 7), 16, 8); err == nil {
		t.Fatalf("expected invalid scales error")
	}
	badG := make([]int32, 16)
	badG[3] = 1
	if _, err := UploadQuantWeight(make([]int32, 16), badG, make([]float32, 8), 16, 8); err == nil {
		t.Fatalf("expected bad group index error")
	}
}

func TestQuantDispatchMalformedDoesNotPanic(t *testing.T) {
	out := NewDevBuf(1)
	x := NewDevBufFrom([]float32{1})
	GemvQ4(out, x, nil)
	GemmQ4(out, x, nil, 2)
	GemvQ4OrGemm(out, x, nil, 2)
}
