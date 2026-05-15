package gpu

import "testing"

func TestDevBufToCPUDownloadFailureKeepsGPUAuthoritative(t *testing.T) {
	b := &DevBuf{
		cpu: []float32{1, 2},
		gpu: &Buffer{Size: 4}, // too small for the CPU slice; Download fails before CUDA call.
		n:   2,
		dev: GPU_DEVICE,
	}
	b.ToCPU()
	if b.dev != GPU_DEVICE {
		t.Fatalf("ToCPU marked CPU authoritative after failed download: dev=%v", b.dev)
	}
}
