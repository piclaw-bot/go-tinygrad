package gpu

// Vulkan compute buffer and shader management.
//
// VkBuf: device-local buffer with host-visible staging
// VkShader: compiled SPIR-V compute pipeline

import (
	"fmt"
	"unsafe"
)

// VkBuf is a Vulkan compute buffer (device-local + host-visible staging).
type VkBuf struct {
	buf    VkBuffer
	mem    VkDeviceMemory
	size   uint64
	mapped unsafe.Pointer
}

// VkBufAlloc allocates a Vulkan buffer accessible from both host and device.
func VkBufAlloc(sizeBytes int) (*VkBuf, error) {
	if !vkReady {
		return nil, fmt.Errorf("vulkan not initialized")
	}

	bufInfo := struct {
		sType                 uint32
		pNext                 uintptr
		flags                 uint32
		size                  uint64
		usage                 uint32
		sharingMode           uint32
		queueFamilyIndexCount uint32
		pQueueFamilyIndices   uintptr
	}{
		sType:       VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		size:        uint64(sizeBytes),
		usage:       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		sharingMode: VK_SHARING_MODE_EXCLUSIVE,
	}

	var buf VkBuffer
	if r := vkCreateBuffer(vkDevice, unsafe.Pointer(&bufInfo), nil, &buf); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateBuffer: %d", r)
	}

	// Get memory requirements
	type memReqs struct {
		size           uint64
		alignment      uint64
		memoryTypeBits uint32
	}
	var reqs memReqs
	vkGetBufferMemoryRequirements(vkDevice, buf, unsafe.Pointer(&reqs))

	// Find host-visible + host-coherent memory type
	type memType struct {
		propertyFlags uint32
		heapIndex     uint32
	}
	type memProps struct {
		memoryTypeCount uint32
		memoryTypes     [32]memType
		memoryHeapCount uint32
		memoryHeaps     [16]struct{ size, flags uint64 }
	}
	var props memProps
	vkGetPhysicalDeviceMemoryProperties(vkPhysDev, unsafe.Pointer(&props))

	memTypeIdx := uint32(0xFFFFFFFF)
	wantFlags := uint32(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
	for i := uint32(0); i < props.memoryTypeCount; i++ {
		if reqs.memoryTypeBits&(1<<i) != 0 && props.memoryTypes[i].propertyFlags&wantFlags == wantFlags {
			memTypeIdx = i
			break
		}
	}
	if memTypeIdx == 0xFFFFFFFF {
		vkDestroyBuffer(vkDevice, buf, nil)
		return nil, fmt.Errorf("no suitable memory type")
	}

	allocInfo := struct {
		sType           uint32
		pNext           uintptr
		allocationSize  uint64
		memoryTypeIndex uint32
	}{
		sType:           VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		allocationSize:  reqs.size,
		memoryTypeIndex: memTypeIdx,
	}

	var mem VkDeviceMemory
	if r := vkAllocateMemory(vkDevice, unsafe.Pointer(&allocInfo), nil, &mem); r != VK_SUCCESS {
		vkDestroyBuffer(vkDevice, buf, nil)
		return nil, fmt.Errorf("vkAllocateMemory: %d", r)
	}

	if r := vkBindBufferMemory(vkDevice, buf, mem, 0); r != VK_SUCCESS {
		vkFreeMemory(vkDevice, mem, nil)
		vkDestroyBuffer(vkDevice, buf, nil)
		return nil, fmt.Errorf("vkBindBufferMemory: %d", r)
	}

	// Map memory
	var mapped unsafe.Pointer
	if r := vkMapMemory(vkDevice, mem, 0, uint64(sizeBytes), 0, &mapped); r != VK_SUCCESS {
		vkFreeMemory(vkDevice, mem, nil)
		vkDestroyBuffer(vkDevice, buf, nil)
		return nil, fmt.Errorf("vkMapMemory: %d", r)
	}

	return &VkBuf{buf: buf, mem: mem, size: uint64(sizeBytes), mapped: mapped}, nil
}

// Upload copies float32 data to the buffer.
func (b *VkBuf) Upload(data []float32) {
	if len(data) == 0 {
		return
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
	dst := unsafe.Slice((*byte)(b.mapped), len(data)*4)
	copy(dst, src)
}

// Download copies float32 data from the buffer.
func (b *VkBuf) Download(data []float32) {
	if len(data) == 0 {
		return
	}
	src := unsafe.Slice((*byte)(b.mapped), len(data)*4)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
	copy(dst, src)
}

// Free releases the buffer and its memory.
func (b *VkBuf) Free() {
	if b == nil {
		return
	}
	if b.mapped != nil {
		vkUnmapMemory(vkDevice, b.mem)
		b.mapped = nil
	}
	if b.buf != 0 {
		vkDestroyBuffer(vkDevice, b.buf, nil)
		b.buf = 0
	}
	if b.mem != 0 {
		vkFreeMemory(vkDevice, b.mem, nil)
		b.mem = 0
	}
}
