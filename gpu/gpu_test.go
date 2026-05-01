package gpu


import (
	"fmt"
	"os"
	"testing"
	"time"
)

func TestNVDirect(t *testing.T) {
	if os.Getenv("NV_IOCTL_TEST") == "" { t.Skip("set NV_IOCTL_TEST=1 to run NV ioctl tests") }
	dev, err := NVInit()
	if err != nil {
		t.Skipf("NV direct not available: %v", err)
	}
	// Don't close — singleton, needed by other tests
	t.Logf("NV device initialized via direct ioctl")
	t.Logf("GPU UUID: %x", dev.gpuUUID)
}

func TestNVZZMemory(t *testing.T) {
	if os.Getenv("NV_IOCTL_TEST") == "" { t.Skip("set NV_IOCTL_TEST=1 to run NV ioctl tests") }
	t.Skip("NV_ESC_RM_ALLOC_MEMORY corrupts RM session in container — run in isolation")
	dev, err := NVInit()
	if err != nil {
		t.Skipf("NV direct not available: %v", err)
	}
	defer dev.Close()

	buf, err := dev.AllocHostMem(4096)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer buf.Free()

	// Write and read back
	data := []float32{42.0, 3.14, 2.718, 1.414}
	if err := buf.Upload(data); err != nil {
		t.Fatal(err)
	}

	out := make([]float32, 4)
	if err := buf.Download(out); err != nil {
		t.Fatal(err)
	}

	for i, v := range out {
		if v != data[i] {
			t.Fatalf("out[%d]=%v want %v", i, v, data[i])
		}
	}
	t.Logf("NV host memory: write/read OK (hMemory=0x%x)", buf.hMemory)
}

func TestGPUInit(t *testing.T) {
	if !Available() {
		t.Skip("GPU not available (no libcuda.so.1)")
	}
	t.Logf("GPU: %s (%d SMs)", DeviceName(), SMCount())
}

func TestGPUMalloc(t *testing.T) {
	if !SgemmReady() { t.Skip("no GPU SGEMM") }
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
		t.Skipf("GPU alloc limit: %v", err)
	}
	elapsed := time.Since(start)
	gflops := 2.0 * float64(N) * float64(N) * float64(N) / elapsed.Seconds() / 1e9
	t.Logf("GPU SGEMM %dx%d: %v (%.1f GFLOPS)", N, N, elapsed, gflops)

	// Try 4096×4096 (may fail if GPU memory is limited)
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
		t.Logf("4096x4096 skipped (GPU memory limit): %v", err)
		return
	}
	elapsed = time.Since(start)
	gflops = 2.0 * float64(N) * float64(N) * float64(N) / elapsed.Seconds() / 1e9
	fmt.Printf("[gpu] SGEMM %dx%d: %v (%.1f GFLOPS)\n", N, N, elapsed, gflops)
	t.Logf("GPU SGEMM %dx%d: %v (%.1f GFLOPS)", N, N, elapsed, gflops)
}

func TestNVZMemoryLarge(t *testing.T) {
	if os.Getenv("NV_IOCTL_TEST") == "" { t.Skip("set NV_IOCTL_TEST=1 to run NV ioctl tests") }
	t.Skip("Merged into TestNVMemory — NVIDIA driver allows one NV_ESC_RM_ALLOC_MEMORY per fd")
	dev, err := NVInit()
	if err != nil {
		t.Skipf("NV direct not available: %v", err)
	}

	size := 1024 * 1024
	buf, err := dev.AllocHostMem(uint64(size * 4))
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer buf.Free()

	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.001
	}
	if err := buf.Upload(data); err != nil {
		t.Fatal(err)
	}

	out := make([]float32, size)
	if err := buf.Download(out); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < size; i += size / 10 {
		if out[i] != data[i] {
			t.Fatalf("out[%d]=%v want %v", i, out[i], data[i])
		}
	}
	t.Logf("NV 4MB host memory: write/read OK (%d floats)", size)
}

func TestNVGPUInfo(t *testing.T) {
	if os.Getenv("NV_IOCTL_TEST") == "" { t.Skip("set NV_IOCTL_TEST=1 to run NV ioctl tests") }
	dev, err := NVInit()
	if err != nil {
		t.Skipf("NV direct not available: %v", err)
	}

	info, err := dev.QueryGPUInfo()
	if err != nil {
		t.Fatalf("query: %v", err)
	}

	t.Logf("GPU: %s (%d SMs = %d GPCs × %d TPC/GPC × %d SM/TPC)",
		info.Arch, info.TotalSMs, info.NumGPCs, info.NumTPCPerGPC, info.NumSMPerTPC)
	t.Logf("Max warps/SM: %d", info.MaxWarpsPerSM)
	t.Logf("Classes: compute=0x%x, dma=0x%x, gpfifo=0x%x",
		info.ComputeClass, info.DMAClass, info.GPFifoClass)

	if info.TotalSMs == 0 {
		t.Fatal("expected >0 SMs")
	}
	if info.Arch == "" {
		t.Fatal("expected arch string")
	}
}

func TestNVChannelGroup(t *testing.T) {
	if os.Getenv("NV_IOCTL_TEST") == "" { t.Skip("set NV_IOCTL_TEST=1 to run NV ioctl tests") }
	dev, err := NVInit()
	if err != nil {
		t.Skipf("NV direct not available: %v", err)
	}

	cg, err := dev.SetupChannelGroup()
	if err != nil {
		t.Fatalf("channel group: %v", err)
	}
	t.Logf("Channel group: handle=0x%x", cg.handle)

	// Try context share
	ctxShare, err := dev.SetupContextShare(cg, dev.vaspace)
	if err != nil {
		t.Logf("Context share (expected to fail without full vaspace): %v", err)
	} else {
		t.Logf("Context share: handle=0x%x", ctxShare)
	}
}
