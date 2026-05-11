#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer Gate { float gate[]; };
layout(set=0, binding=1) buffer Up { float up[]; };
layout(push_constant) uniform P { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    float x = gate[i];
    float x3 = x * x * x;
    float z = 0.7978845608 * (x + 0.044715 * x3);
    float ez = exp(2.0 * z);
    float tanh_z = (ez - 1.0) / (ez + 1.0);
    gate[i] = 0.5 * x * (1.0 + tanh_z) * up[i];
}
