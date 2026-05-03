package gpu

import (
	"math"
	"testing"
)

func TestCompilerSiLUMul(t *testing.T) {
	if !SgemmReady() {
		t.Skip("no GPU")
	}

	spec := FusedSiLUMulSpec()
	k, err := Compile(spec)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	n := 1024
	a := NewDevBuf(n)
	b := NewDevBuf(n)
	out := NewDevBuf(n)
	for i := range a.Data() {
		a.Data()[i] = float32(i)*0.01 - 5.0
		b.Data()[i] = float32(i)*0.005 + 0.5
	}

	// CPU reference
	cpuOut := make([]float32, n)
	for i := 0; i < n; i++ {
		x := a.Data()[i]
		s := x / (1.0 + float32(math.Exp(float64(-x))))
		cpuOut[i] = s * b.Data()[i]
	}

	// GPU via compiled kernel
	a.ToGPU()
	b.ToGPU()
	out.ToGPU()
	k.Launch(n, a.GPUPtr(), b.GPUPtr(), out.GPUPtr())
	Sync()
	gpuOut := out.Data()

	maxDiff := float32(0)
	for i := 0; i < n; i++ {
		d := float32(math.Abs(float64(gpuOut[i] - cpuOut[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("JIT SiLU*Mul %d: maxDiff=%e", n, maxDiff)
	if maxDiff > 0.01 {
		t.Fatalf("drift: %e (gpu[0]=%v cpu[0]=%v)", maxDiff, gpuOut[0], cpuOut[0])
	}
}

func TestCompilerAdd(t *testing.T) {
	if !SgemmReady() {
		t.Skip("no GPU")
	}

	// Use batch-compiled kernel
	InitAllKernels()
	k := jitAdd
	if k == nil {
		t.Skip("JIT Add kernel not available")
	}

	n := 512
	a := NewDevBuf(n)
	b := NewDevBuf(n)
	out := NewDevBuf(n)
	for i := range a.Data() {
		a.Data()[i] = float32(i) * 0.1
		b.Data()[i] = float32(n-i) * 0.1
	}

	a.ToGPU()
	b.ToGPU()
	out.ToGPU()
	k.Launch(n, a.GPUPtr(), b.GPUPtr(), out.GPUPtr())
	Sync()

	d := out.Data()
	for i := 0; i < n; i++ {
		want := a.cpu[i] + b.cpu[i]
		if math.Abs(float64(d[i]-want)) > 0.001 {
			t.Fatalf("add[%d]=%v want %v", i, d[i], want)
		}
	}
	t.Log("JIT Add OK")
}

func TestCompilerCache(t *testing.T) {
	if !SgemmReady() {
		t.Skip("no GPU")
	}

	// Compile same spec twice — should hit cache
	spec1 := FusedSiLUMulSpec()
	k1, _ := Compile(spec1)

	spec2 := FusedSiLUMulSpec()
	k2, _ := Compile(spec2)

	if k1 != k2 {
		t.Fatal("cache miss for identical spec")
	}
	t.Log("kernel cache: OK")
}
