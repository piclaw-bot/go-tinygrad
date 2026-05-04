package gpu

// SPIR-V compute shaders for Vulkan inference.
//
// Pre-compiled SPIR-V binary embedded as Go byte slices.
// Generated from GLSL compute shaders:
//
// The SPIR-V binary format is stable and works on all Vulkan implementations.
// We embed the compiled SPIR-V directly (no runtime compilation needed).
//
// To regenerate: glslangValidator -V shader.comp -o shader.spv
//
// For now, we hand-assemble minimal SPIR-V for a vector add shader
// to prove the pipeline works. Production shaders will be compiled from GLSL.

import (
	"encoding/binary"
	"fmt"
	"unsafe"
)

// VkComputeShader wraps a compiled SPIR-V compute pipeline.
type VkComputeShader struct {
	pipeline       VkPipeline
	pipelineLayout VkPipelineLayout
	descSetLayout  VkDescriptorSetLayout
	descPool       VkDescriptorPool
	numBuffers     int
}

// assembleSPIRV creates a minimal SPIR-V module for a compute shader.
// This is a helper that builds SPIR-V binary from a GLSL-like specification.
// For production, use pre-compiled SPIR-V from glslangValidator.
func assembleSPIRV(glslSource string) ([]byte, error) {
	// For now, return pre-built SPIR-V for common shaders
	switch glslSource {
	case "vec_add":
		return spirvVecAdd, nil
	case "gemv_f32":
		return spirvGemvF32, nil
	default:
		return nil, fmt.Errorf("unknown shader: %s", glslSource)
	}
}

// Pre-compiled SPIR-V for vector add:
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer A { float a[]; };
// layout(set=0, binding=1) buffer B { float b[]; };
// layout(set=0, binding=2) buffer C { float c[]; };
// layout(push_constant) uniform Params { uint n; };
// void main() {
//     uint i = gl_GlobalInvocationID.x;
//     if (i < n) c[i] = a[i] + b[i];
// }
var spirvVecAdd = buildSPIRVVecAdd()

// Pre-compiled SPIR-V for F32 GEMV (matrix-vector multiply)
var spirvGemvF32 = buildSPIRVGemvF32()

func buildSPIRVVecAdd() []byte {
	// Minimal SPIR-V 1.0 compute shader: c[i] = a[i] + b[i]
	// This is the binary encoding of the GLSL above.
	// Hand-assembled for portability (no glslang dependency).
	var b []byte
	w := func(words ...uint32) {
		for _, word := range words {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], word)
			b = append(b, buf[:]...)
		}
	}

	// SPIR-V header
	w(0x07230203) // Magic
	w(0x00010000) // Version 1.0
	w(0x00000000) // Generator
	w(50)         // Bound (max ID + 1)
	w(0)          // Schema

	// Capability Shader
	w(0x00020011, 1)
	// Memory model: Logical GLSL450
	w(0x0003000E, 0, 1)
	// Entry point: GLCompute, main (ID 4), gl_GlobalInvocationID (ID 10)
	w(0x0006000F, 5, 4, 0x6E69616D, 0x00000000, 10) // "main"
	// Execution mode: LocalSize 256, 1, 1
	w(0x00060010, 4, 17, 256, 1, 1)

	// Decorations
	w(0x00040047, 10, 11, 28) // BuiltIn GlobalInvocationID
	// Descriptor set/binding for buffers A(0,0), B(0,1), C(0,2)
	w(0x00040047, 20, 34, 0) // DescriptorSet 0
	w(0x00040047, 20, 33, 0) // Binding 0
	w(0x00040047, 25, 34, 0)
	w(0x00040047, 25, 33, 1)
	w(0x00040047, 30, 34, 0)
	w(0x00040047, 30, 33, 2)
	// ArrayStride for runtime arrays
	w(0x00040047, 18, 6, 4)
	w(0x00040047, 23, 6, 4)
	w(0x00040047, 28, 6, 4)
	// Block decoration
	w(0x00030047, 19, 2)
	w(0x00040048, 19, 0, 35, 0) // Offset 0
	w(0x00030047, 24, 2)
	w(0x00040048, 24, 0, 35, 0)
	w(0x00030047, 29, 2)
	w(0x00040048, 29, 0, 35, 0)

	// Types
	w(0x00020013, 2)     // void
	w(0x00030021, 3, 2)  // function type void()
	w(0x00040015, 6, 32, 0) // uint32
	w(0x00040017, 7, 6, 3)  // uvec3
	w(0x00040020, 8, 1, 7)  // ptr<Input, uvec3>
	w(0x0004003B, 8, 10, 1) // gl_GlobalInvocationID
	w(0x0004002B, 6, 11, 0) // const 0

	// Float type
	w(0x00030016, 15, 32) // float32
	// Runtime array of float
	w(0x0003001D, 18, 15)
	// Struct { float[] }
	w(0x0003001E, 19, 18)
	// Pointer to struct (StorageBuffer)
	w(0x00040020, 21, 12, 19)
	// Variable A
	w(0x0004003B, 21, 20, 12)

	// Same for B and C
	w(0x0003001D, 23, 15)
	w(0x0003001E, 24, 23)
	w(0x00040020, 26, 12, 24)
	w(0x0004003B, 26, 25, 12)

	w(0x0003001D, 28, 15)
	w(0x0003001E, 29, 28)
	w(0x00040020, 31, 12, 29)
	w(0x0004003B, 31, 30, 12)

	// Pointer to float (StorageBuffer)
	w(0x00040020, 35, 12, 15)
	// Bool type
	w(0x00020014, 40)

	// Push constant: struct { uint n; }
	w(0x0003001E, 42, 6)           // struct { uint }
	w(0x00040020, 43, 9, 42)      // ptr<PushConstant>
	w(0x0004003B, 43, 44, 9)      // variable
	w(0x00040020, 45, 9, 6)       // ptr<PushConstant, uint>
	w(0x00040047, 42, 2)          // Block
	w(0x00040048, 42, 0, 35, 0)   // Offset 0

	// Function main
	w(0x00050036, 2, 4, 0, 3) // OpFunction void main
	w(0x000200F8, 5)          // OpLabel

	// %12 = AccessChain gl_GlobalInvocationID[0] → uint
	w(0x00040020, 9, 1, 6)   // ptr<Input, uint>
	w(0x00050041, 9, 12, 10, 11) // AccessChain
	w(0x0004003D, 6, 13, 12)     // Load uint

	// Load n from push constant
	w(0x00050041, 45, 46, 44, 11) // AccessChain
	w(0x0004003D, 6, 47, 46)      // Load n

	// if (i < n)
	w(0x0005008B, 40, 48, 13, 47) // ULessThan
	w(0x000300F7, 49, 0)           // SelectionMerge
	w(0x000400FA, 48, 50, 49)     // BranchConditional

	// True block
	w(0x000200F8, 50) // OpLabel

	// a[i]
	w(0x00050041, 35, 36, 20, 11, 13) // AccessChain A.data[i]
	w(0x0004003D, 15, 37, 36)          // Load float

	// b[i]
	w(0x00050041, 35, 38, 25, 11, 13)
	w(0x0004003D, 15, 39, 38)

	// c[i] = a[i] + b[i]
	w(0x00050081, 15, 41, 37, 39)       // FAdd
	w(0x00050041, 35, 32, 30, 11, 13)   // AccessChain C.data[i]
	w(0x0003003E, 32, 41)                // Store

	w(0x000200F9, 49) // Branch to merge
	w(0x000200F8, 49) // Merge label
	w(0x000100FD)      // Return
	w(0x00010038)      // FunctionEnd

	return b
}

func buildSPIRVGemvF32() []byte {
	// Placeholder — full GEMV shader is complex.
	// For now return vec_add SPIR-V (will be replaced with actual GEMV).
	return buildSPIRVVecAdd()
}

// LoadSPIRV creates a Vulkan compute pipeline from SPIR-V bytecode.
func LoadSPIRV(spirv []byte, numBuffers int) (*VkComputeShader, error) {
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

	// Create descriptor set layout (N storage buffers)
	bindings := make([]struct {
		binding         uint32
		descriptorType  uint32
		descriptorCount uint32
		stageFlags      uint32
		pSamplers       uintptr
	}, numBuffers)
	for i := range bindings {
		bindings[i] = struct {
			binding         uint32
			descriptorType  uint32
			descriptorCount uint32
			stageFlags      uint32
			pSamplers       uintptr
		}{
			binding:         uint32(i),
			descriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			descriptorCount: 1,
			stageFlags:      0x20, // VK_SHADER_STAGE_COMPUTE_BIT
		}
	}

	layoutInfo := struct {
		sType        uint32
		pNext        uintptr
		flags        uint32
		bindingCount uint32
		pBindings    unsafe.Pointer
	}{
		sType:        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		bindingCount: uint32(numBuffers),
		pBindings:    unsafe.Pointer(&bindings[0]),
	}

	var descSetLayout VkDescriptorSetLayout
	if r := vkCreateDescriptorSetLayout(vkDevice, unsafe.Pointer(&layoutInfo), nil, &descSetLayout); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", r)
	}

	// Create pipeline layout
	plInfo := struct {
		sType          uint32
		pNext          uintptr
		flags          uint32
		setLayoutCount uint32
		pSetLayouts    unsafe.Pointer
		pushRangeCount uint32
		pPushRanges    uintptr
	}{
		sType:          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount: 1,
		pSetLayouts:    unsafe.Pointer(&descSetLayout),
	}

	var pipelineLayout VkPipelineLayout
	if r := vkCreatePipelineLayout(vkDevice, unsafe.Pointer(&plInfo), nil, &pipelineLayout); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", r)
	}

	// Create compute pipeline
	entryName := append([]byte("main"), 0)
	stageInfo := struct {
		sType               uint32
		pNext               uintptr
		flags               uint32
		stage               uint32
		module              VkShaderModule
		pName               unsafe.Pointer
		pSpecializationInfo uintptr
	}{
		sType:  0x12, // VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
		stage:  0x20, // VK_SHADER_STAGE_COMPUTE_BIT
		module: shaderModule,
		pName:  unsafe.Pointer(&entryName[0]),
	}

	pipelineInfo := struct {
		sType              uint32
		pNext              uintptr
		flags              uint32
		stage              [48]byte // inline PipelineShaderStageCreateInfo
		layout             VkPipelineLayout
		basePipelineHandle uintptr
		basePipelineIndex  int32
	}{
		sType:  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		layout: pipelineLayout,
	}
	// Copy stage info into inline field
	copy(pipelineInfo.stage[:], (*[48]byte)(unsafe.Pointer(&stageInfo))[:])

	var pipeline VkPipeline
	if r := vkCreateComputePipelines(vkDevice, 0, 1, unsafe.Pointer(&pipelineInfo), nil, &pipeline); r != VK_SUCCESS {
		return nil, fmt.Errorf("vkCreateComputePipelines: %d", r)
	}

	return &VkComputeShader{
		pipeline:       pipeline,
		pipelineLayout: pipelineLayout,
		descSetLayout:  descSetLayout,
		numBuffers:     numBuffers,
	}, nil
}
