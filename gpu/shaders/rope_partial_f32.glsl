#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer Q { float q[]; };
layout(set=0, binding=1) buffer Cos { float cos_table[]; };
layout(set=0, binding=2) buffer Sin { float sin_table[]; };
layout(push_constant) uniform P { uint headDim; uint rotHalf; uint nHeads; uint pos; };
void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint head = tid / headDim;
    uint d = tid % headDim;
    if (head >= nHeads) return;
    if (d >= rotHalf * 2) return; // only rotate first rotHalf pairs
    uint hf = d % rotHalf;
    float c = cos_table[pos * rotHalf + hf];
    float s = sin_table[pos * rotHalf + hf];
    uint base = head * headDim;
    float q0 = q[base + hf];
    float q1 = q[base + hf + rotHalf];
    if (d < rotHalf) {
        q[base + d] = q0 * c - q1 * s;
    } else {
        q[base + d] = q0 * s + q1 * c;
    }
}
