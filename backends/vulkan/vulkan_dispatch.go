package vulkan

// Vulkan compute dispatch: command buffers, descriptor binding, shader execution.
//
// This completes the Vulkan compute pipeline:
//   VkComputeKernel: compiled shader + pipeline + descriptor layout
//   VkDispatch: record command buffer → bind descriptors → dispatch → submit → wait
//
// The pattern for each operation:
//   1. Bind pipeline
//   2. Update descriptor set with buffer bindings
//   3. Push constants (dimensions, eps, etc.)
//   4. Dispatch workgroups
//   5. Submit + fence wait

import (
	"fmt"
	"unsafe"
)

// VkComputeKernel is a ready-to-dispatch Vulkan compute shader.
type VkComputeKernel struct {
	pipeline       VkPipeline
	pipelineLayout VkPipelineLayout
	descSetLayout  VkDescriptorSetLayout
	descPool       VkDescriptorPool
	descSet        VkDescriptorSet
	cmdBuf         VkCommandBuffer
	fence          VkFence
	numBuffers     int
	pushSize       int
}

// VkKernelCreate builds a compute kernel from SPIR-V with N buffer bindings and push constant size.
func VkKernelCreate(spirv []byte, numBuffers int, pushConstantSize int) (*VkComputeKernel, error) {
	if !vkReady {
		return nil, fmt.Errorf("vulkan not initialized")
	}

	// Create shader module
	moduleInfo := struct {
		sType    uint32
		pNext    uintptr
		flags    uint32
		codeSize uint64
		pCode    unsafe.Pointer
	}{
		sType:    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		codeSize: uint64(len(spirv)),
		pCode:    unsafe.Pointer(&spirv[0]),
	}
	var shaderModule VkShaderModule
	if r := vkCreateShaderModule(vkDevice, unsafe.Pointer(&moduleInfo), nil, &shaderModule); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateShaderModule: %d", r)
	}

	// Descriptor set layout: N storage buffers
	type descBinding struct {
		binding, descType, descCount, stageFlags uint32
		pSamplers                                uintptr
	}
	bindings := make([]descBinding, numBuffers)
	for i := range bindings {
		bindings[i] = descBinding{
			binding:    uint32(i),
			descType:   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			descCount:  1,
			stageFlags: 0x20, // COMPUTE
		}
	}
	layoutInfo := struct {
		sType, pNext, flags uint32
		bindingCount        uint32
		pBindings           unsafe.Pointer
	}{
		sType:        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		bindingCount: uint32(numBuffers),
		pBindings:    unsafe.Pointer(&bindings[0]),
	}
	var descSetLayout VkDescriptorSetLayout
	if r := vkCreateDescriptorSetLayout(vkDevice, unsafe.Pointer(&layoutInfo), nil, &descSetLayout); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", r)
	}

	// Push constant range
	pushRange := struct {
		stageFlags uint32
		offset     uint32
		size       uint32
	}{stageFlags: 0x20, size: uint32(pushConstantSize)}

	// Pipeline layout with push constants
	plInfo := struct {
		sType, pNext, flags    uint32
		setLayoutCount         uint32
		pSetLayouts            unsafe.Pointer
		pushConstantRangeCount uint32
		pPushConstantRanges    unsafe.Pointer
	}{
		sType:                  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount:         1,
		pSetLayouts:            unsafe.Pointer(&descSetLayout),
		pushConstantRangeCount: 1,
		pPushConstantRanges:    unsafe.Pointer(&pushRange),
	}
	if pushConstantSize == 0 {
		plInfo.pushConstantRangeCount = 0
		plInfo.pPushConstantRanges = nil
	}
	var pipelineLayout VkPipelineLayout
	if r := vkCreatePipelineLayout(vkDevice, unsafe.Pointer(&plInfo), nil, &pipelineLayout); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", r)
	}

	// Compute pipeline
	entryName := append([]byte("main"), 0)
	type stageCI struct {
		sType, pNext, flags, stage uint32
		module                     VkShaderModule
		pName                      unsafe.Pointer
		pSpec                      uintptr
	}
	stage := stageCI{
		sType:  0x12, // PIPELINE_SHADER_STAGE_CREATE_INFO
		stage:  0x20, // COMPUTE
		module: shaderModule,
		pName:  unsafe.Pointer(&entryName[0]),
	}
	type computePCI struct {
		sType, pNext, flags uint32
		stage               stageCI
		layout              VkPipelineLayout
		basePH              uintptr
		basePI              int32
	}
	pci := computePCI{
		sType:  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		stage:  stage,
		layout: pipelineLayout,
	}
	var pipeline VkPipeline
	if r := vkCreateComputePipelines(vkDevice, 0, 1, unsafe.Pointer(&pci), nil, &pipeline); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateComputePipelines: %d", r)
	}

	// Descriptor pool
	poolSize := struct {
		descType  uint32
		descCount uint32
	}{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uint32(numBuffers)}
	poolInfo := struct {
		sType, pNext, flags uint32
		maxSets             uint32
		poolSizeCount       uint32
		pPoolSizes          unsafe.Pointer
	}{
		sType:         VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		maxSets:       1,
		poolSizeCount: 1,
		pPoolSizes:    unsafe.Pointer(&poolSize),
	}
	var descPool VkDescriptorPool
	if r := vkCreateDescriptorPool(vkDevice, unsafe.Pointer(&poolInfo), nil, &descPool); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", r)
	}

	// Allocate descriptor set
	allocInfo := struct {
		sType, pNext   uint32
		descriptorPool VkDescriptorPool
		descSetCount   uint32
		pSetLayouts    unsafe.Pointer
	}{
		sType:          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool: descPool,
		descSetCount:   1,
		pSetLayouts:    unsafe.Pointer(&descSetLayout),
	}
	var descSet VkDescriptorSet
	if r := vkAllocateDescriptorSets(vkDevice, unsafe.Pointer(&allocInfo), &descSet); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", r)
	}

	// Allocate command buffer
	cmdAllocInfo := struct {
		sType, pNext    uint32
		commandPool     VkCommandPool
		level           uint32
		commandBufCount uint32
	}{
		sType:           VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool:     vkCmdPool,
		level:           VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		commandBufCount: 1,
	}
	var cmdBuf VkCommandBuffer
	if r := vkAllocateCommandBuffers(vkDevice, unsafe.Pointer(&cmdAllocInfo), &cmdBuf); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %d", r)
	}

	// Create fence
	fenceInfo := struct {
		sType, pNext, flags uint32
	}{sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}
	var fence VkFence
	if r := vkCreateFence(vkDevice, unsafe.Pointer(&fenceInfo), nil, &fence); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateFence: %d", r)
	}

	return &VkComputeKernel{
		pipeline:       pipeline,
		pipelineLayout: pipelineLayout,
		descSetLayout:  descSetLayout,
		descPool:       descPool,
		descSet:        descSet,
		cmdBuf:         cmdBuf,
		fence:          fence,
		numBuffers:     numBuffers,
		pushSize:       pushConstantSize,
	}, nil
}

// Dispatch executes the compute kernel with given buffers and workgroup dimensions.
func (k *VkComputeKernel) Dispatch(groupsX, groupsY, groupsZ uint32, bufs []*VkBuf, pushData unsafe.Pointer) error {
	// Update descriptor set with buffer bindings
	type bufInfo struct {
		buffer VkBuffer
		offset uint64
		rng    uint64 // VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF
	}
	type writeDS struct {
		sType, pNext     uint32
		dstSet           VkDescriptorSet
		dstBinding       uint32
		dstArrayElement  uint32
		descriptorCount  uint32
		descriptorType   uint32
		pImageInfo       uintptr
		pBufferInfo      unsafe.Pointer
		pTexelBufferView uintptr
	}
	writes := make([]writeDS, len(bufs))
	bufInfos := make([]bufInfo, len(bufs))
	for i, buf := range bufs {
		bufInfos[i] = bufInfo{buffer: buf.buf, rng: 0xFFFFFFFFFFFFFFFF}
		writes[i] = writeDS{
			sType:           VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			dstSet:          k.descSet,
			dstBinding:      uint32(i),
			descriptorCount: 1,
			descriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			pBufferInfo:     unsafe.Pointer(&bufInfos[i]),
		}
	}
	vkUpdateDescriptorSets(vkDevice, uint32(len(writes)), unsafe.Pointer(&writes[0]), 0, nil)

	// Record command buffer
	beginInfo := struct {
		sType, pNext, flags uint32
	}{sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}
	if r := vkBeginCommandBuffer(k.cmdBuf, unsafe.Pointer(&beginInfo)); r != VK_SUCCESS {
		return fmt.Errorf("vkBeginCommandBuffer: %d", r)
	}

	vkCmdBindPipeline(k.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline)
	vkCmdBindDescriptorSets(k.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout, 0, 1, &k.descSet, 0, nil)

	// Push constants if any
	if k.pushSize > 0 && pushData != nil {
		vkCmdPushConstants(k.cmdBuf, k.pipelineLayout, 0x20, 0, uint32(k.pushSize), pushData)
	}

	vkCmdDispatch(k.cmdBuf, groupsX, groupsY, groupsZ)

	if r := vkEndCommandBuffer(k.cmdBuf); r != VK_SUCCESS {
		return fmt.Errorf("vkEndCommandBuffer: %d", r)
	}

	// Submit
	submitInfo := struct {
		sType, pNext                       uint32
		waitSemaphoreCount                 uint32
		pWaitSemaphores, pWaitDstStageMask uintptr
		commandBufferCount                 uint32
		pCommandBuffers                    unsafe.Pointer
		signalSemaphoreCount               uint32
		pSignalSemaphores                  uintptr
	}{
		sType:              VK_STRUCTURE_TYPE_SUBMIT_INFO,
		commandBufferCount: 1,
		pCommandBuffers:    unsafe.Pointer(&k.cmdBuf),
	}
	vkResetFences(vkDevice, 1, &k.fence)
	if r := vkQueueSubmit(vkQueue, 1, unsafe.Pointer(&submitInfo), k.fence); r != VK_SUCCESS {
		return fmt.Errorf("vkQueueSubmit: %d", r)
	}

	// Wait for completion (1 second timeout)
	if r := vkWaitForFences(vkDevice, 1, &k.fence, 1, 1_000_000_000); r != VK_SUCCESS {
		return fmt.Errorf("vkWaitForFences: %d", r)
	}

	return nil
}

// vkCmdPushConstants — needs to be registered
var vkCmdPushConstants func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer)
