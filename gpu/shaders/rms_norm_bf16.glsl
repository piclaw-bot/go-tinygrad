#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer X { uint x[]; };  // BF16 in lower 16 bits
layout(set=0, binding=1) buffer W { uint w[]; };
layout(push_constant) uniform P { uint n; float eps; };
shared float sdata[256];
float bf16_to_f32(uint v) { return uintBitsToFloat(v << 16); }
uint f32_to_bf16(float f) { return floatBitsToUint(f) >> 16; }
void main() {
    uint tid = gl_LocalInvocationID.x;
    float ss = 0.0;
    for (uint i = tid; i < n; i += 256) {
        float v = bf16_to_f32(x[i]);
        ss += v * v;
    }
    sdata[tid] = ss;
    barrier();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float invRMS = inversesqrt(sdata[0] / float(n) + eps);
    barrier();
    for (uint i = tid; i < n; i += 256)
        x[i] = f32_to_bf16(bf16_to_f32(w[i]) * bf16_to_f32(x[i]) * invRMS);
}
