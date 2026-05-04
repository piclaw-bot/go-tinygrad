// Vulkan compute shaders for go-pherence inference.
// Compile with: glslangValidator -V -S comp <file>.glsl -o <file>.spv
// Then use: go run scripts/embed-spirv.go to update Go byte slices.
//
// All shaders use workgroup size 256 and storage buffers.
// BF16 is emulated via uint32 bitshift (no extensions required).

// ========== vec_add_f32.glsl ==========
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer A { float a[]; };
// layout(set=0, binding=1) buffer B { float b[]; };
// layout(set=0, binding=2) buffer C { float c[]; };
// layout(push_constant) uniform P { uint n; };
// void main() {
//     uint i = gl_GlobalInvocationID.x;
//     if (i < n) c[i] = a[i] + b[i];
// }

// ========== vec_add_bf16.glsl ==========
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer A { uint a[]; };  // 2× BF16 packed per uint
// layout(set=0, binding=1) buffer B { uint b[]; };
// layout(set=0, binding=2) buffer C { uint c[]; };
// layout(push_constant) uniform P { uint n; };  // n = number of uint pairs
// void main() {
//     uint i = gl_GlobalInvocationID.x;
//     if (i >= n) return;
//     uint pa = a[i], pb = b[i];
//     float a0 = uintBitsToFloat(pa << 16);
//     float a1 = uintBitsToFloat(pa & 0xFFFF0000u);
//     float b0 = uintBitsToFloat(pb << 16);
//     float b1 = uintBitsToFloat(pb & 0xFFFF0000u);
//     c[i] = (floatBitsToUint(a0 + b0) >> 16) | (floatBitsToUint(a1 + b1) & 0xFFFF0000u);
// }

// ========== rms_norm_f32.glsl ==========
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer X { float x[]; };
// layout(set=0, binding=1) buffer W { float w[]; };
// layout(push_constant) uniform P { uint n; float eps; };
// shared float sdata[256];
// void main() {
//     uint tid = gl_LocalInvocationID.x;
//     float ss = 0.0;
//     for (uint i = tid; i < n; i += 256) ss += x[i] * x[i];
//     sdata[tid] = ss;
//     barrier();
//     for (uint s = 128; s > 0; s >>= 1) {
//         if (tid < s) sdata[tid] += sdata[tid + s];
//         barrier();
//     }
//     float invRMS = inversesqrt(sdata[0] / float(n) + eps);
//     barrier();
//     for (uint i = tid; i < n; i += 256) x[i] = w[i] * x[i] * invRMS;
// }

// ========== rms_norm_bf16.glsl ==========
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer X { uint x[]; };  // BF16 in lower 16 bits
// layout(set=0, binding=1) buffer W { uint w[]; };
// layout(push_constant) uniform P { uint n; float eps; };
// shared float sdata[256];
// float bf16_to_f32(uint v) { return uintBitsToFloat(v << 16); }
// uint f32_to_bf16(float f) { return floatBitsToUint(f) >> 16; }
// void main() {
//     uint tid = gl_LocalInvocationID.x;
//     float ss = 0.0;
//     for (uint i = tid; i < n; i += 256) {
//         float v = bf16_to_f32(x[i]);
//         ss += v * v;
//     }
//     sdata[tid] = ss;
//     barrier();
//     for (uint s = 128; s > 0; s >>= 1) {
//         if (tid < s) sdata[tid] += sdata[tid + s];
//         barrier();
//     }
//     float invRMS = inversesqrt(sdata[0] / float(n) + eps);
//     barrier();
//     for (uint i = tid; i < n; i += 256)
//         x[i] = f32_to_bf16(bf16_to_f32(w[i]) * bf16_to_f32(x[i]) * invRMS);
// }

// ========== gemv_f32.glsl ==========
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer X { float x[]; };
// layout(set=0, binding=1) buffer W { float w[]; };  // [outDim * inDim]
// layout(set=0, binding=2) buffer OUT { float out_buf[]; };
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
//     if (tid == 0) out_buf[row] = sdata[0];
// }

// ========== gemv_bf16_mixed.glsl ==========
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer X { uint x[]; };    // BF16 activations
// layout(set=0, binding=1) buffer W { float w[]; };   // F32 weights
// layout(set=0, binding=2) buffer OUT { uint out_buf[]; }; // BF16 output
// layout(push_constant) uniform P { uint inDim; uint outDim; };
// shared float sdata[256];
// void main() {
//     uint row = gl_WorkGroupID.x;
//     uint tid = gl_LocalInvocationID.x;
//     if (row >= outDim) return;
//     float sum = 0.0;
//     for (uint i = tid; i < inDim; i += 256)
//         sum += w[row * inDim + i] * uintBitsToFloat(x[i] << 16);
//     sdata[tid] = sum;
//     barrier();
//     for (uint s = 128; s > 0; s >>= 1) {
//         if (tid < s) sdata[tid] += sdata[tid + s];
//         barrier();
//     }
//     if (tid == 0) out_buf[row] = floatBitsToUint(sdata[0]) >> 16;
// }

// ========== silu_mul_f32.glsl ==========
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer GATE { float gate[]; };
// layout(set=0, binding=1) buffer UP { float up[]; };
// layout(set=0, binding=2) buffer OUT { float out_buf[]; };
// layout(push_constant) uniform P { uint n; };
// void main() {
//     uint i = gl_GlobalInvocationID.x;
//     if (i >= n) return;
//     float g = gate[i];
//     float s = g / (1.0 + exp(-g));  // SiLU
//     out_buf[i] = s * up[i];
// }

// ========== attention_score.glsl ==========
// #version 450
// layout(local_size_x = 256) in;
// layout(set=0, binding=0) buffer Q { float q[]; };       // [numHeads * headDim]
// layout(set=0, binding=1) buffer K { float k_cache[]; }; // [seqLen * kvDim]
// layout(set=0, binding=2) buffer S { float scores[]; };  // [numHeads * seqLen]
// layout(push_constant) uniform P {
//     uint numHeads; uint numKVHeads; uint headDim; uint seqLen;
// };
// void main() {
//     // Each workgroup handles one (head, time) pair
//     uint head = gl_WorkGroupID.x;
//     uint t = gl_WorkGroupID.y;
//     uint tid = gl_LocalInvocationID.x;
//     if (head >= numHeads || t >= seqLen) return;
//     uint kvHead = head / (numHeads / numKVHeads);
//     float scale = inversesqrt(float(headDim));
//     // Dot product Q[head] · K[t]
//     float sum = 0.0;
//     for (uint d = tid; d < headDim; d += 256)
//         sum += q[head * headDim + d] * k_cache[t * numKVHeads * headDim + kvHead * headDim + d];
//     // Shared memory reduce (omitted for brevity — same pattern as GEMV)
//     // scores[head * seqLen + t] = sum * scale;
// }

package gpu
// This file contains GLSL source as documentation.
// The actual SPIR-V binaries are in vulkan_spirv.go.
// To compile: glslangValidator -V -S comp <shader>.glsl -o <shader>.spv
