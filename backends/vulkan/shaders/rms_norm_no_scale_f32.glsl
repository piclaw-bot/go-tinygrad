#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer X { float x[]; };
layout(set=0, binding=1) buffer Out { float out_data[]; };
layout(push_constant) uniform P { uint n; float eps; };
shared float sdata[256];
void main() {
    uint tid = gl_LocalInvocationID.x;
    float sum = 0.0;
    for (uint i = tid; i < n; i += 256) sum += x[i] * x[i];
    sdata[tid] = sum;
    barrier();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float invRMS = inversesqrt(sdata[0] / float(n) + eps);
    for (uint i = tid; i < n; i += 256) out_data[i] = x[i] * invRMS;
}
