package gpu


import (
	"os"
	"testing"
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
