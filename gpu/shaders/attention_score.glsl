#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer Q { float q[]; };
layout(set=0, binding=1) buffer K { float k_cache[]; };
layout(set=0, binding=2) buffer S { float scores[]; };
layout(push_constant) uniform P {
    uint numHeads; uint numKVHeads; uint headDim; uint seqLen; float scale;
};
shared float sdata[256];
void main() {
    uint head = gl_WorkGroupID.x;
    uint t = gl_WorkGroupID.y;
    uint tid = gl_LocalInvocationID.x;
    if (head >= numHeads || t >= seqLen) return;
    uint kvHead = head / (numHeads / numKVHeads);
    uint kvDim = numKVHeads * headDim;
    float sum = 0.0;
    for (uint d = tid; d < headDim; d += 256)
        sum += q[head * headDim + d] * k_cache[t * kvDim + kvHead * headDim + d];
    sdata[tid] = sum;
    barrier();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0) scores[head * seqLen + t] = sdata[0] * scale;
}
