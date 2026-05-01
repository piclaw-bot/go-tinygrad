package gpu

import (
	"fmt"
	"testing"
	"time"
)

func TestNVDirect(t *testing.T) {
	dev, err := NVInit()
	if err != nil {
		t.Skipf("NV direct not available: %v", err)
	}
	defer dev.Close()
	t.Logf("NV device initialized via direct ioctl")
}

func TestGPUInit(t *testing.T) {
	if !Available() {
		t.Skip("GPU not available (no libcuda.so.1)")
	}
	t.Logf("GPU: %s (%d SMs)", DeviceName(), SMCount())
}

func TestGPUMalloc(t *testing.T) {
	if !Available() {
		t.Skip("GPU not available")
	}

	buf, err := Malloc(1024)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Free()

	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i)
	}
	if err := buf.Upload(data); err != nil {
		t.Fatal(err)
	}

	out := make([]float32, 1024)
	if err := buf.Download(out); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 10; i++ {
		if out[i] != data[i] {
			t.Fatalf("out[%d]=%v want %v", i, out[i], data[i])
		}
	}
	t.Log("GPU malloc + upload + download: OK")
}

func TestGPUSgemm(t *testing.T) {
	if !SgemmReady() {
		t.Skip("GPU SGEMM not available")
	}

	// 2×3 @ 3×2 = 2×2
	A := []float32{1, 2, 3, 4, 5, 6}
	B := []float32{7, 8, 9, 10, 11, 12}

	C, err := SgemmHost(2, 2, 3, 1.0, A, B)
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{58, 64, 139, 154}
	for i, v := range C {
		if v != want[i] {
			t.Fatalf("C[%d]=%v want %v", i, v, want[i])
		}
	}
	t.Logf("GPU SGEMM result: %v ✓", C)
}

func TestGPUSgemmLarge(t *testing.T) {
	if !SgemmReady() {
		t.Skip("GPU SGEMM not available")
	}

	// 1024×1024 SGEMM benchmark
	N := 1024
	A := make([]float32, N*N)
	B := make([]float32, N*N)
	for i := range A {
		A[i] = 0.001 * float32(i%1000)
		B[i] = 0.001 * float32((i*7)%1000)
	}

	start := time.Now()
	_, err := SgemmHost(N, N, N, 1.0, A, B)
	if err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	gflops := 2.0 * float64(N) * float64(N) * float64(N) / elapsed.Seconds() / 1e9
	t.Logf("GPU SGEMM %dx%d: %v (%.1f GFLOPS)", N, N, elapsed, gflops)

	// Try 4096×4096 (closer to real model sizes)
	N = 4096
	A = make([]float32, N*N)
	B = make([]float32, N*N)
	for i := range A {
		A[i] = 0.001 * float32(i%1000)
		B[i] = 0.001 * float32((i*7)%1000)
	}

	start = time.Now()
	_, err = SgemmHost(N, N, N, 1.0, A, B)
	if err != nil {
		t.Fatal(err)
	}
	elapsed = time.Since(start)
	gflops = 2.0 * float64(N) * float64(N) * float64(N) / elapsed.Seconds() / 1e9
	fmt.Printf("[gpu] SGEMM %dx%d: %v (%.1f GFLOPS)\n", N, N, elapsed, gflops)
	t.Logf("GPU SGEMM %dx%d: %v (%.1f GFLOPS)", N, N, elapsed, gflops)
}
