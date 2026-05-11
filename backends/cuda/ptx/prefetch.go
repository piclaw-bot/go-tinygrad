package ptx

// Prefetch kernel PTX: reads 1 float per 128B to warm L2
var PrefetchPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry prefetch_l2(
    .param .u64 buf,
    .param .u32 N
) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f0;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r1, [N];
    setp.ge.u32 %p, %r3, %r1;
    @%p bra done;

    ld.param.u64 %rd0, [buf];
    mul.wide.u32 %rd1, %r3, 128;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f0, [%rd2];
done:
    ret;
}
`
