#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer A { uint a[]; };  // 2× BF16 packed per uint
layout(set=0, binding=1) buffer B { uint b[]; };
layout(set=0, binding=2) buffer C { uint c[]; };
layout(push_constant) uniform P { uint n; };  // n = number of uint pairs
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    uint pa = a[i], pb = b[i];
    float a0 = uintBitsToFloat(pa << 16);
    float a1 = uintBitsToFloat(pa & 0xFFFF0000u);
    float b0 = uintBitsToFloat(pb << 16);
    float b1 = uintBitsToFloat(pb & 0xFFFF0000u);
    c[i] = (floatBitsToUint(a0 + b0) >> 16) | (floatBitsToUint(a1 + b1) & 0xFFFF0000u);
}
