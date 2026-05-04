package gpu

import (
	"testing"
)

func TestVulkanInit(t *testing.T) {
	if !VulkanInit() {
		t.Skip("no Vulkan device")
	}
	t.Logf("Vulkan: %s", VulkanDeviceName())
}

func TestVulkanBuffer(t *testing.T) {
	if !VulkanInit() {
		t.Skip("no Vulkan device")
	}

	// Allocate a buffer
	buf, err := VkBufAlloc(1024 * 4) // 1024 floats
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer buf.Free()

	// Upload data
	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	buf.Upload(data)

	// Download and verify
	out := make([]float32, 1024)
	buf.Download(out)
	for i := range data {
		if out[i] != data[i] {
			t.Fatalf("mismatch at %d: got %f want %f", i, out[i], data[i])
		}
	}
	t.Log("Vulkan buffer upload/download: OK")
}
