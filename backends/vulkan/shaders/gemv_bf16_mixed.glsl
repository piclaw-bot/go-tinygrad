#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer X { uint x[]; };    // BF16 activations
layout(set=0, binding=1) buffer W { float w[]; };   // F32 weights
layout(set=0, binding=2) buffer OUT { uint out_buf[]; }; // BF16 output
layout(push_constant) uniform P { uint inDim; uint outDim; };
shared float sdata[256];
void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    if (row >= outDim) return;
    float sum = 0.0;
    for (uint i = tid; i < inDim; i += 256)
        sum += w[row * inDim + i] * uintBitsToFloat(x[i] << 16);
    sdata[tid] = sum;
    barrier();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0) out_buf[row] = floatBitsToUint(sdata[0]) >> 16;
}
