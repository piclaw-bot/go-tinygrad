package gpu

import "testing"

func TestNVBufferUploadDownloadValidation(t *testing.T) {
	var buf *NVBuffer
	if err := buf.Upload([]float32{1}); err == nil {
		t.Fatal("nil NVBuffer Upload returned nil error")
	}
	if err := buf.Download([]float32{1}); err == nil {
		t.Fatal("nil NVBuffer Download returned nil error")
	}
	buf.Free()

	buf = &NVBuffer{size: 4, cpuAddr: 1, cpuMem: make([]byte, 4)}
	if err := buf.Upload(nil); err != nil {
		t.Fatalf("empty upload should be no-op: %v", err)
	}
	if err := buf.Download(nil); err != nil {
		t.Fatalf("empty download should be no-op: %v", err)
	}
	if err := buf.Upload([]float32{1, 2}); err == nil {
		t.Fatal("oversized upload returned nil error")
	}
	if err := buf.Download(make([]float32, 2)); err == nil {
		t.Fatal("oversized download returned nil error")
	}
	if err := buf.Upload([]float32{3}); err != nil {
		t.Fatalf("valid upload: %v", err)
	}
	out := []float32{0}
	if err := buf.Download(out); err != nil {
		t.Fatalf("valid download: %v", err)
	}
	if out[0] != 3 {
		t.Fatalf("download=%v want 3", out[0])
	}
}

func TestNVDeviceAllocHostMemValidation(t *testing.T) {
	var dev *NVDevice
	if _, err := dev.AllocHostMem(4096); err == nil {
		t.Fatal("nil NVDevice AllocHostMem returned nil error")
	}
	dev = &NVDevice{}
	if _, err := dev.AllocHostMem(0); err == nil {
		t.Fatal("zero AllocHostMem returned nil error")
	}
	if err := dev.mapToCPU(nil); err == nil {
		t.Fatal("mapToCPU accepted nil buffer")
	}
}
