package ptx

// NVFP4DequantF32PTX materializes ModelOpt/NVFP4 weights to row-major F32.
// One CUDA thread decodes one logical weight element from packed E2M1 FP4 plus
// F8_E4M3FN block scale and scalar weight_scale_2.
const NVFP4DequantF32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry nvfp4_dequant_f32(
    .param .u64 W,
    .param .u64 S,
    .param .u64 O,
    .param .f32 SCALE2,
    .param .u32 TOTAL,
    .param .u32 IN_DIM,
    .param .u32 GROUP_SIZE
) {
    .reg .pred %p<8>;
    .reg .u32 %r<40>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<16>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;        // idx
    ld.param.u32 %r4, [TOTAL];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    ld.param.u64 %rd0, [W];
    ld.param.u64 %rd1, [S];
    ld.param.u64 %rd2, [O];
    ld.param.f32 %f0, [SCALE2];
    ld.param.u32 %r5, [IN_DIM];
    ld.param.u32 %r6, [GROUP_SIZE];

    div.u32 %r7, %r3, %r5;                // row
    rem.u32 %r8, %r3, %r5;                // col
    shr.u32 %r9, %r5, 1;                  // packed bytes per row
    mad.lo.u32 %r10, %r7, %r9, 0;         // row byte offset
    shr.u32 %r26, %r8, 1;                 // col/2 packed byte within row
    add.u32 %r10, %r10, %r26;             // weight byte offset
    add.u64 %rd3, %rd0, %r10;
    ld.global.u8 %r11, [%rd3];
    and.b32 %r12, %r8, 1;
    setp.ne.u32 %p1, %r12, 0;
    @%p1 shr.u32 %r11, %r11, 4;
    and.b32 %r11, %r11, 15;               // FP4 code

    and.b32 %r13, %r11, 7;                // magnitude code
    and.b32 %r14, %r11, 8;                // sign bit
    mov.f32 %f1, 0f00000000;
    setp.eq.u32 %p2, %r13, 1; @%p2 mov.f32 %f1, 0f3F000000;
    setp.eq.u32 %p2, %r13, 2; @%p2 mov.f32 %f1, 0f3F800000;
    setp.eq.u32 %p2, %r13, 3; @%p2 mov.f32 %f1, 0f3FC00000;
    setp.eq.u32 %p2, %r13, 4; @%p2 mov.f32 %f1, 0f40000000;
    setp.eq.u32 %p2, %r13, 5; @%p2 mov.f32 %f1, 0f40400000;
    setp.eq.u32 %p2, %r13, 6; @%p2 mov.f32 %f1, 0f40800000;
    setp.eq.u32 %p2, %r13, 7; @%p2 mov.f32 %f1, 0f40C00000;
    setp.ne.u32 %p3, %r14, 0;
    @%p3 neg.f32 %f1, %f1;

    div.u32 %r15, %r8, %r6;               // group
    div.u32 %r16, %r5, %r6;               // groups per row
    mad.lo.u32 %r17, %r7, %r16, %r15;
    add.u64 %rd4, %rd1, %r17;
    ld.global.u8 %r18, [%rd4];            // E4M3FN scale code

    and.b32 %r19, %r18, 127;
    setp.eq.u32 %p4, %r19, 127;
    @%p4 bra scale_nan;
    and.b32 %r20, %r18, 128;              // sign
    shr.u32 %r21, %r18, 3;
    and.b32 %r21, %r21, 15;               // exp
    and.b32 %r22, %r18, 7;                // mant
    setp.eq.u32 %p5, %r21, 0;
    @%p5 bra scale_subnormal;

    add.u32 %r23, %r21, 120;              // exp - 7 + 127
    shl.b32 %r23, %r23, 23;
    shl.b32 %r24, %r22, 20;
    or.b32 %r25, %r23, %r24;
    setp.ne.u32 %p6, %r20, 0;
    @%p6 or.b32 %r25, %r25, 2147483648;
    mov.b32 %f2, %r25;
    bra scale_done;

scale_subnormal:
    cvt.rn.f32.u32 %f2, %r22;
    mul.f32 %f2, %f2, 0f3B000000;         // mant * 2^-9
    setp.ne.u32 %p6, %r20, 0;
    @%p6 neg.f32 %f2, %f2;
    bra scale_done;

scale_nan:
    mov.f32 %f2, 0f7FC00000;

scale_done:
    mul.f32 %f3, %f1, %f2;
    mul.f32 %f3, %f3, %f0;
    mul.wide.u32 %rd5, %r3, 4;
    add.u64 %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f3;

done:
    ret;
}
`
