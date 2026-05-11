#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer GATE { float gate[]; };
layout(set=0, binding=1) buffer UP { float up[]; };
layout(set=0, binding=2) buffer OUT { float out_buf[]; };
layout(push_constant) uniform P { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    float g = gate[i];
    float s = g / (1.0 + exp(-g));  // SiLU
    out_buf[i] = s * up[i];
}
