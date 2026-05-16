package gpu

import "testing"

func TestDevBufToGPURejectsCPUAuthoritativeNilCPU(t *testing.T) {
	b := &DevBuf{
		gpu: &Buffer{Ptr: 1, Size: 4},
		n:   1,
		dev: CPU,
	}
	if err := b.ToGPU(); err == nil {
		t.Fatal("ToGPU accepted CPU-authoritative buffer with nil CPU backing")
	}
	if b.dev != CPU {
		t.Fatalf("ToGPU changed authoritative device after failed upload: %v", b.dev)
	}
}

func TestDevBufGPUPtrRejectsStaleGPUWhenCPUAuthoritative(t *testing.T) {
	b := &DevBuf{
		gpu: &Buffer{Ptr: 1, Size: 4},
		n:   1,
		dev: CPU,
	}
	if got := b.GPUPtr(); got != nil {
		t.Fatalf("GPUPtr returned stale GPU pointer for CPU-authoritative nil-CPU buffer: %#v", got)
	}
	if b.dev != CPU {
		t.Fatalf("GPUPtr changed authoritative device after failed upload: %v", b.dev)
	}
}
