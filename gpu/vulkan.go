package gpu

// Vulkan compute backend for inference on any GPU.
//
// Uses purego to dlopen libvulkan.so (no CGo required).
// Loads pre-compiled SPIR-V compute shaders for:
//   - GEMV (quantized and float)
//   - RMSNorm
//   - Vector operations (add, mul, scale, silu)
//   - Attention
//
// Architecture:
//   1. Instance → PhysicalDevice → LogicalDevice → Queue
//   2. Allocate device-local memory for weights
//   3. Create compute pipelines from SPIR-V modules
//   4. Record command buffers with dispatch commands
//   5. Submit and fence-wait for results
//
// This is the portable GPU path — works on:
//   - NVIDIA (desktop + Jetson)
//   - Intel (iGPU + Arc)
//   - AMD (RDNA)
//   - Qualcomm Adreno (Android/Linux)
//   - Apple (via MoltenVK)
//   - ARM Mali

import (
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Vulkan types
type VkInstance uintptr
type VkPhysicalDevice uintptr
type VkDevice uintptr
type VkQueue uintptr
type VkCommandPool uintptr
type VkCommandBuffer uintptr
type VkBuffer uintptr
type VkDeviceMemory uintptr
type VkPipeline uintptr
type VkPipelineLayout uintptr
type VkShaderModule uintptr
type VkDescriptorSetLayout uintptr
type VkDescriptorPool uintptr
type VkDescriptorSet uintptr
type VkFence uintptr
type VkResult int32

const (
	VK_SUCCESS                                          VkResult = 0
	VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO                       = 1
	VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO                         = 3
	VK_STRUCTURE_TYPE_SUBMIT_INFO                                = 4
	VK_STRUCTURE_TYPE_FENCE_CREATE_INFO                          = 8
	VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO                         = 12
	VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO                  = 16
	VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO               = 29
	VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO                = 30
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO          = 32
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO               = 34
	VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET                       = 35
	VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO                = 33
	VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO                   = 39
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO               = 40
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO                  = 42
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO                       = 5

	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT          = 0x00000020
	VK_BUFFER_USAGE_TRANSFER_SRC_BIT            = 0x00000001
	VK_BUFFER_USAGE_TRANSFER_DST_BIT            = 0x00000002
	VK_SHARING_MODE_EXCLUSIVE                   = 0
	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT         = 0x00000001
	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT         = 0x00000002
	VK_MEMORY_PROPERTY_HOST_COHERENT_BIT        = 0x00000004
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER           = 7
	VK_PIPELINE_BIND_POINT_COMPUTE              = 1
	VK_COMMAND_BUFFER_LEVEL_PRIMARY             = 0
	VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
	VK_QUEUE_COMPUTE_BIT                        = 0x00000002
	VK_PHYSICAL_DEVICE_TYPE_OTHER               = 0
	VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU      = 1
	VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU        = 2
	VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU         = 3
	VK_PHYSICAL_DEVICE_TYPE_CPU                 = 4

	VK_NULL_HANDLE = 0
)

// Vulkan state
var (
	vkLib                uintptr
	vkInstance           VkInstance
	vkPhysDev            VkPhysicalDevice
	vkDevice             VkDevice
	vkQueue              VkQueue
	vkCmdPool            VkCommandPool
	vkComputeQueueFamily uint32
	vkReady              bool
	vkDevName            string
)

// Vulkan function pointers
var (
	vkCreateInstance                         func(unsafe.Pointer, unsafe.Pointer, *VkInstance) VkResult
	vkEnumeratePhysicalDevices               func(VkInstance, *uint32, *VkPhysicalDevice) VkResult
	vkGetPhysicalDeviceProperties            func(VkPhysicalDevice, unsafe.Pointer)
	vkGetPhysicalDeviceMemoryProperties      func(VkPhysicalDevice, unsafe.Pointer)
	vkGetPhysicalDeviceQueueFamilyProperties func(VkPhysicalDevice, *uint32, unsafe.Pointer)
	vkCreateDevice                           func(VkPhysicalDevice, unsafe.Pointer, unsafe.Pointer, *VkDevice) VkResult
	vkGetDeviceQueue                         func(VkDevice, uint32, uint32, *VkQueue)
	vkCreateCommandPool                      func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkCommandPool) VkResult
	vkCreateBuffer                           func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkBuffer) VkResult
	vkAllocateMemory                         func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkDeviceMemory) VkResult
	vkBindBufferMemory                       func(VkDevice, VkBuffer, VkDeviceMemory, uint64) VkResult
	vkMapMemory                              func(VkDevice, VkDeviceMemory, uint64, uint64, uint32, *unsafe.Pointer) VkResult
	vkUnmapMemory                            func(VkDevice, VkDeviceMemory)
	vkCreateShaderModule                     func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkShaderModule) VkResult
	vkCreateComputePipelines                 func(VkDevice, uintptr, uint32, unsafe.Pointer, unsafe.Pointer, *VkPipeline) VkResult
	vkCreatePipelineLayout                   func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkPipelineLayout) VkResult
	vkCreateDescriptorSetLayout              func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkDescriptorSetLayout) VkResult
	vkCreateDescriptorPool                   func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkDescriptorPool) VkResult
	vkAllocateDescriptorSets                 func(VkDevice, unsafe.Pointer, *VkDescriptorSet) VkResult
	vkUpdateDescriptorSets                   func(VkDevice, uint32, unsafe.Pointer, uint32, unsafe.Pointer)
	vkAllocateCommandBuffers                 func(VkDevice, unsafe.Pointer, *VkCommandBuffer) VkResult
	vkBeginCommandBuffer                     func(VkCommandBuffer, unsafe.Pointer) VkResult
	vkEndCommandBuffer                       func(VkCommandBuffer) VkResult
	vkCmdBindPipeline                        func(VkCommandBuffer, uint32, VkPipeline)
	vkCmdBindDescriptorSets                  func(VkCommandBuffer, uint32, VkPipelineLayout, uint32, uint32, *VkDescriptorSet, uint32, unsafe.Pointer)
	vkCmdDispatch                            func(VkCommandBuffer, uint32, uint32, uint32)
	vkQueueSubmit                            func(VkQueue, uint32, unsafe.Pointer, VkFence) VkResult
	vkQueueWaitIdle                          func(VkQueue) VkResult
	vkCreateFence                            func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkFence) VkResult
	vkWaitForFences                          func(VkDevice, uint32, *VkFence, uint32, uint64) VkResult
	vkResetFences                            func(VkDevice, uint32, *VkFence) VkResult
	vkGetBufferMemoryRequirements            func(VkDevice, VkBuffer, unsafe.Pointer)
	vkDestroyBuffer                          func(VkDevice, VkBuffer, unsafe.Pointer)
	vkFreeMemory                             func(VkDevice, VkDeviceMemory, unsafe.Pointer)
)

// VulkanInit initializes the Vulkan compute backend.
func VulkanInit() bool {
	if vkReady {
		return true
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	lib, err := purego.Dlopen("libvulkan.so.1", purego.RTLD_LAZY)
	if err != nil {
		lib, err = purego.Dlopen("libvulkan.so", purego.RTLD_LAZY)
		if err != nil {
			return false
		}
	}
	vkLib = lib

	// Register all function pointers
	reg := func(fptr interface{}, name string) {
		func() {
			defer func() { recover() }()
			purego.RegisterLibFunc(fptr, lib, name)
		}()
	}

	reg(&vkCreateInstance, "vkCreateInstance")
	reg(&vkEnumeratePhysicalDevices, "vkEnumeratePhysicalDevices")
	reg(&vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties")
	reg(&vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties")
	reg(&vkGetPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties")
	reg(&vkCreateDevice, "vkCreateDevice")
	reg(&vkGetDeviceQueue, "vkGetDeviceQueue")
	reg(&vkCreateCommandPool, "vkCreateCommandPool")
	reg(&vkCreateBuffer, "vkCreateBuffer")
	reg(&vkAllocateMemory, "vkAllocateMemory")
	reg(&vkBindBufferMemory, "vkBindBufferMemory")
	reg(&vkMapMemory, "vkMapMemory")
	reg(&vkUnmapMemory, "vkUnmapMemory")
	reg(&vkCreateShaderModule, "vkCreateShaderModule")
	reg(&vkCreateComputePipelines, "vkCreateComputePipelines")
	reg(&vkCreatePipelineLayout, "vkCreatePipelineLayout")
	reg(&vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout")
	reg(&vkCreateDescriptorPool, "vkCreateDescriptorPool")
	reg(&vkAllocateDescriptorSets, "vkAllocateDescriptorSets")
	reg(&vkUpdateDescriptorSets, "vkUpdateDescriptorSets")
	reg(&vkAllocateCommandBuffers, "vkAllocateCommandBuffers")
	reg(&vkBeginCommandBuffer, "vkBeginCommandBuffer")
	reg(&vkEndCommandBuffer, "vkEndCommandBuffer")
	reg(&vkCmdBindPipeline, "vkCmdBindPipeline")
	reg(&vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets")
	reg(&vkCmdDispatch, "vkCmdDispatch")
	reg(&vkQueueSubmit, "vkQueueSubmit")
	reg(&vkQueueWaitIdle, "vkQueueWaitIdle")
	reg(&vkCreateFence, "vkCreateFence")
	reg(&vkWaitForFences, "vkWaitForFences")
	reg(&vkResetFences, "vkResetFences")
	reg(&vkCmdPushConstants, "vkCmdPushConstants")
	reg(&vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements")
	reg(&vkDestroyBuffer, "vkDestroyBuffer")
	reg(&vkFreeMemory, "vkFreeMemory")

	if vkCreateInstance == nil {
		return false
	}

	// Create instance (no extensions needed for compute-only)
	appInfo := struct {
		sType              uint32
		pNext              uintptr
		pApplicationName   *byte
		applicationVersion uint32
		pEngineName        *byte
		engineVersion      uint32
		apiVersion         uint32
	}{
		sType:      0,                     // VK_STRUCTURE_TYPE_APPLICATION_INFO
		apiVersion: (1 << 22) | (3 << 12), // Vulkan 1.3
	}

	createInfo := struct {
		sType                   uint32
		pNext                   uintptr
		flags                   uint32
		pApplicationInfo        unsafe.Pointer
		enabledLayerCount       uint32
		ppEnabledLayerNames     uintptr
		enabledExtensionCount   uint32
		ppEnabledExtensionNames uintptr
	}{
		sType:            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		pApplicationInfo: unsafe.Pointer(&appInfo),
	}

	if r := vkCreateInstance(unsafe.Pointer(&createInfo), nil, &vkInstance); r != VK_SUCCESS {
		fmt.Printf("[vulkan] vkCreateInstance failed: %d\n", r)
		return false
	}

	// Enumerate physical devices
	var devCount uint32
	vkEnumeratePhysicalDevices(vkInstance, &devCount, nil)
	if devCount == 0 {
		fmt.Println("[vulkan] no physical devices found")
		return false
	}

	devs := make([]VkPhysicalDevice, devCount)
	vkEnumeratePhysicalDevices(vkInstance, &devCount, &devs[0])

	// Pick best compute device. CPU/software Vulkan implementations (e.g. llvmpipe)
	// are not an inference backend and are rejected by default; allow them only for
	// explicit shader debugging with GO_PHERENCE_VULKAN_ALLOW_CPU=1.
	type devProps struct {
		apiVersion    uint32
		driverVersion uint32
		vendorID      uint32
		deviceID      uint32
		deviceType    uint32
		deviceName    [256]byte
		_padding      [1024]byte // pipelineCacheUUID + limits + sparseProperties (avoid stack overwrite)
	}

	allowCPU := os.Getenv("GO_PHERENCE_VULKAN_ALLOW_CPU") == "1"
	bestIdx := -1
	bestPriority := uint32(999)
	bestName := ""
	for i := uint32(0); i < devCount; i++ {
		var props devProps
		vkGetPhysicalDeviceProperties(devs[i], unsafe.Pointer(&props))
		name := string(props.deviceName[:])
		for j, b := range props.deviceName {
			if b == 0 {
				name = string(props.deviceName[:j])
				break
			}
		}

		priority := uint32(999)
		switch props.deviceType {
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			priority = 0
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			priority = 1
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			priority = 2
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			if allowCPU {
				priority = 10
			}
		case VK_PHYSICAL_DEVICE_TYPE_OTHER:
			if allowCPU {
				priority = 11
			}
		}
		if priority < bestPriority {
			bestIdx = int(i)
			bestPriority = priority
			bestName = name
		}
	}
	if bestIdx < 0 {
		fmt.Println("[vulkan] no non-CPU Vulkan GPU found (set GO_PHERENCE_VULKAN_ALLOW_CPU=1 to allow software/CPU drivers)")
		return false
	}
	vkPhysDev = devs[bestIdx]
	vkDevName = bestName

	// Find compute queue family
	var queueCount uint32
	vkGetPhysicalDeviceQueueFamilyProperties(vkPhysDev, &queueCount, nil)

	type queueFamilyProps struct {
		queueFlags                  uint32
		queueCount                  uint32
		timestampValidBits          uint32
		minImageTransferGranularity [3]uint32
	}
	queueFams := make([]queueFamilyProps, queueCount)
	vkGetPhysicalDeviceQueueFamilyProperties(vkPhysDev, &queueCount, unsafe.Pointer(&queueFams[0]))

	found := false
	for i := uint32(0); i < queueCount; i++ {
		if queueFams[i].queueFlags&VK_QUEUE_COMPUTE_BIT != 0 {
			vkComputeQueueFamily = i
			found = true
			break
		}
	}
	if !found {
		fmt.Println("[vulkan] no compute queue family found")
		return false
	}

	// Create logical device with one compute queue
	queuePriority := float32(1.0)
	queueCreateInfo := struct {
		sType            uint32
		pNext            uintptr
		flags            uint32
		queueFamilyIndex uint32
		queueCount       uint32
		pQueuePriorities unsafe.Pointer
	}{
		sType:            0x02, // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
		queueFamilyIndex: vkComputeQueueFamily,
		queueCount:       1,
		pQueuePriorities: unsafe.Pointer(&queuePriority),
	}

	deviceCreateInfo := struct {
		sType                   uint32
		pNext                   uintptr
		flags                   uint32
		queueCreateInfoCount    uint32
		pQueueCreateInfos       unsafe.Pointer
		enabledLayerCount       uint32
		ppEnabledLayerNames     uintptr
		enabledExtensionCount   uint32
		ppEnabledExtensionNames uintptr
		pEnabledFeatures        uintptr
	}{
		sType:                VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		queueCreateInfoCount: 1,
		pQueueCreateInfos:    unsafe.Pointer(&queueCreateInfo),
	}

	if r := vkCreateDevice(vkPhysDev, unsafe.Pointer(&deviceCreateInfo), nil, &vkDevice); r != VK_SUCCESS {
		fmt.Printf("[vulkan] vkCreateDevice failed: %d\n", r)
		return false
	}

	vkGetDeviceQueue(vkDevice, vkComputeQueueFamily, 0, &vkQueue)

	// Create command pool
	poolInfo := struct {
		sType            uint32
		pNext            uintptr
		flags            uint32
		queueFamilyIndex uint32
	}{
		sType:            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		flags:            0x02, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
		queueFamilyIndex: vkComputeQueueFamily,
	}

	if r := vkCreateCommandPool(vkDevice, unsafe.Pointer(&poolInfo), nil, &vkCmdPool); r != VK_SUCCESS {
		fmt.Printf("[vulkan] vkCreateCommandPool failed: %d\n", r)
		return false
	}

	vkReady = true
	fmt.Printf("[vulkan] %s — compute ready\n", vkDevName)
	return true
}

// VulkanReady returns true if Vulkan compute is available.
func VulkanReady() bool { return vkReady }

// VulkanDeviceName returns the Vulkan device name.
func VulkanDeviceName() string { return vkDevName }
