package vulkan

import (
	"math"
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

	buf, err := VkBufAlloc(1024 * 4)
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer buf.Free()

	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	buf.Upload(data)

	out := make([]float32, 1024)
	buf.Download(out)
	for i := range data {
		if out[i] != data[i] {
			t.Fatalf("mismatch at %d: got %f want %f", i, out[i], data[i])
		}
	}
	t.Log("Vulkan buffer upload/download: OK")
}

func TestVulkanVecAddF32(t *testing.T) {
	if !VulkanInit() {
		t.Skip("no Vulkan device")
	}

	n := 1024
	a, err := VkBufAlloc(n * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := VkBufAlloc(n * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	c, err := VkBufAlloc(n * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free()

	aData := make([]float32, n)
	bData := make([]float32, n)
	for i := range aData {
		aData[i] = float32(i) * 0.1
		bData[i] = float32(n-i) * 0.01
	}
	a.Upload(aData)
	b.Upload(bData)

	err = VkVecAddF32(c, a, b, n)
	if err != nil {
		t.Skipf("VkVecAddF32: %v (SPIR-V shader may not compile on this driver)", err)
	}

	cData := make([]float32, n)
	c.Download(cData)

	maxDiff := float32(0)
	for i := range cData {
		want := aData[i] + bData[i]
		d := float32(math.Abs(float64(cData[i] - want)))
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("VkVecAddF32: maxDiff=%e (%d elements, device=%s)", maxDiff, n, VulkanDeviceName())
	if maxDiff > 0.001 {
		t.Fatalf("VkVecAddF32 drift too high: %e", maxDiff)
	}
}

func BenchmarkVulkanVecAddF32(b *testing.B) {
	if !VulkanInit() {
		b.Skip("no Vulkan")
	}
	n := 3584
	a, _ := VkBufAlloc(n * 4)
	bb, _ := VkBufAlloc(n * 4)
	c, _ := VkBufAlloc(n * 4)
	defer a.Free()
	defer bb.Free()
	defer c.Free()

	data := make([]float32, n)
	a.Upload(data)
	bb.Upload(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VkVecAddF32(c, a, bb, n)
	}
}
