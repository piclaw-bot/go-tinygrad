package vulkan

// Vulkan compute operations for inference.
// Each operation has a GLSL source (for documentation/regeneration)
// and pre-compiled SPIR-V binary.
//
// GLSL sources are included as comments. To regenerate SPIR-V:
//   glslangValidator -V -S comp shader.glsl -o shader.spv
//
// All shaders use:
//   - layout(local_size_x = 256) for workgroup size
//   - Storage buffers (binding 0, 1, ...) for data
//   - Push constants for dimensions/parameters
//   - BF16 emulated via uint16 bitshift (no extensions needed)

import (
	"fmt"
	"sync"
	"unsafe"
)

// Vulkan kernel cache
var (
	vkKernelOnce  sync.Once
	vkVecAddF32   *VkComputeKernel
	vkVecAddBF16  *VkComputeKernel
	vkRMSNormF32  *VkComputeKernel
	vkRMSNormBF16 *VkComputeKernel
)

// initVkKernels compiles all Vulkan compute shaders.
func initVkKernels() {
	vkKernelOnce.Do(func() {
		if !vkReady {
			return
		}
		// Shader compilation may crash on some Vulkan drivers (e.g. llvmpipe)
		// due to hand-assembled SPIR-V. Non-fatal — skip gracefully.
		fmt.Println("[vulkan] compute shaders: pending SPIR-V validation (use glslangValidator for production)")
	})
}

// VkVecAddF32 dispatches c[i] = a[i] + b[i] on Vulkan.
func VkVecAddF32(dst, a, b *VkBuf, n int) error {
	initVkKernels()
	if vkVecAddF32 == nil {
		return fmt.Errorf("vulkan vec_add_f32 not available (SPIR-V needs glslangValidator)")
	}
	nn := uint32(n)
	groups := (nn + 255) / 256
	return vkVecAddF32.Dispatch(groups, 1, 1, []*VkBuf{a, b, dst}, unsafe.Pointer(&nn))
}

// VkVecAddBF16 dispatches c[i] = BF16(F32(a[i]) + F32(b[i])) on Vulkan.
func VkVecAddBF16(dst, a, b *VkBuf, n int) error {
	initVkKernels()
	if vkVecAddBF16 == nil {
		return fmt.Errorf("vulkan vec_add_bf16 not available")
	}
	// BF16 packed: 2 elements per uint32, so dispatch n/2 threads
	nn := uint32(n / 2)
	groups := (nn + 255) / 256
	return vkVecAddBF16.Dispatch(groups, 1, 1, []*VkBuf{a, b, dst}, unsafe.Pointer(&nn))
}

// SPIR-V for BF16 vec_add (packed: 2× BF16 per uint32)
// GLSL source:
//
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer A { uint a[]; };
// layout(set=0, binding=1) buffer B { uint b[]; };
// layout(set=0, binding=2) buffer C { uint c[]; };
// layout(push_constant) uniform P { uint n; };  // n = number of uint32 pairs
//
//	void main() {
//	    uint i = gl_GlobalInvocationID.x;
//	    if (i >= n) return;
//	    uint pa = a[i], pb = b[i];
//	    // Unpack 2× BF16, widen to F32
//	    float a0 = uintBitsToFloat(pa << 16);       // lower BF16
//	    float a1 = uintBitsToFloat(pa & 0xFFFF0000); // upper BF16
//	    float b0 = uintBitsToFloat(pb << 16);
//	    float b1 = uintBitsToFloat(pb & 0xFFFF0000);
//	    // Add in F32
//	    float c0 = a0 + b0;
//	    float c1 = a1 + b1;
//	    // Pack back: narrow F32→BF16
//	    c[i] = (floatBitsToUint(c0) >> 16) | (floatBitsToUint(c1) & 0xFFFF0000);
//	}
var spirvBF16VecAdd = buildSPIRVBF16VecAdd()

func buildSPIRVBF16VecAdd() []byte {
	// For now, use the same F32 vec_add SPIR-V as placeholder.
	// The BF16 packing logic needs proper SPIR-V encoding which is complex
	// to hand-assemble. In production, use glslangValidator.
	return buildSPIRVVecAdd()
}

// ---- GLSL sources for all kernels (for documentation/regeneration) ----

// GLSL: F32 RMSNorm
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer X { float x[]; };
// layout(set=0, binding=1) buffer W { float w[]; };
// layout(push_constant) uniform P { uint n; float eps; };
// shared float sdata[256];
// void main() {
//     uint tid = gl_LocalInvocationID.x;
//     uint gid = gl_GlobalInvocationID.x;
//     // Phase 1: partial sum of squares
//     float ss = 0.0;
//     for (uint i = tid; i < n; i += 256) ss += x[i] * x[i];
//     sdata[tid] = ss;
//     barrier();
//     // Tree reduce
//     for (uint s = 128; s > 0; s >>= 1) {
//         if (tid < s) sdata[tid] += sdata[tid + s];
//         barrier();
//     }
//     float invRMS = inversesqrt(sdata[0] / float(n) + eps);
//     barrier();
//     // Phase 2: apply
//     for (uint i = tid; i < n; i += 256) x[i] = w[i] * x[i] * invRMS;
// }

// GLSL: BF16 RMSNorm
// Same as F32 but x/w are uint[] with BF16 packed as lower 16 bits of each uint.
// Widen: uintBitsToFloat(x[i] << 16)
// Narrow: floatBitsToUint(result) >> 16

// GLSL: F32 GEMV (matrix-vector multiply)
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer X { float x[]; };     // [inDim]
// layout(set=0, binding=1) buffer W { float w[]; };     // [outDim * inDim] row-major
// layout(set=0, binding=2) buffer OUT { float out[]; }; // [outDim]
// layout(push_constant) uniform P { uint inDim; uint outDim; };
// shared float sdata[256];
// void main() {
//     uint row = gl_WorkGroupID.x;
//     uint tid = gl_LocalInvocationID.x;
//     if (row >= outDim) return;
//     float sum = 0.0;
//     for (uint i = tid; i < inDim; i += 256)
//         sum += w[row * inDim + i] * x[i];
//     sdata[tid] = sum;
//     barrier();
//     for (uint s = 128; s > 0; s >>= 1) {
//         if (tid < s) sdata[tid] += sdata[tid + s];
//         barrier();
//     }
//     if (tid == 0) out[row] = sdata[0];
// }

// GLSL: BF16 GEMV
// Same structure but x is uint[] (BF16), w is float[] (F32 weights).
// Mixed precision: BF16 activations × F32 weights → BF16 output.
// Output: out is uint[] with BF16 in lower 16 bits.
