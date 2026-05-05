#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer A { float a[]; };
layout(set=0, binding=1) buffer B { float b[]; };
layout(set=0, binding=2) buffer C { float c[]; };
layout(push_constant) uniform P { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < n) c[i] = a[i] + b[i];
}
