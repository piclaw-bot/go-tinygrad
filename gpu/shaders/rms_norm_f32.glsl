#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer X { float x[]; };
layout(set=0, binding=1) buffer W { float w[]; };
layout(push_constant) uniform P { uint n; float eps; };
shared float sdata[256];
void main() {
    uint tid = gl_LocalInvocationID.x;
    float ss = 0.0;
    for (uint i = tid; i < n; i += 256) ss += x[i] * x[i];
    sdata[tid] = ss;
    barrier();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float invRMS = inversesqrt(sdata[0] / float(n) + eps);
    barrier();
    for (uint i = tid; i < n; i += 256) x[i] = w[i] * x[i] * invRMS;
}
