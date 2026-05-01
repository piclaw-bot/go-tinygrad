#include "textflag.h"

// gebpMicroKernel: 4×16 GEBP tile, ARM64 NEON
// C[0:4, 0:16] += alpha * A[0:4, 0:k] × Bp[0:k, 0:16]
//
// Args (56 bytes): k:0  alpha:8  a:16  lda:24  bp:32  c:40  ldc:48
//
// FMUL/FADD vectors encoded via WORD macros (not in Go's arm64 asm).
// FMUL Vd.4S, Vn.4S, Vm.4S: 0x6e20dc00 | (Vm<<16) | (Vn<<5) | Vd
// FADD Vd.4S, Vn.4S, Vm.4S: 0x4e20d400 | (Vm<<16) | (Vn<<5) | Vd

#define FMUL_V20(vn, vd)       WORD $(0x6e34dc00 | ((vn)<<5) | (vd))
#define VFADD(vm, vn, vd)      WORD $(0x4e20d400 | ((vm)<<16) | ((vn)<<5) | (vd))

TEXT ·gebpMicroKernel(SB), NOSPLIT, $16-56
    MOVD    k+0(FP), R0
    FMOVS   alpha+8(FP), F31
    MOVD    a+16(FP), R1
    MOVD    lda+24(FP), R2
    MOVD    bp+32(FP), R3
    MOVD    c+40(FP), R4
    MOVD    ldc+48(FP), R5

    MOVD    R4, 0(RSP)
    LSL     $2, R5, R5
    MOVD    R5, 8(RSP)
    LSL     $2, R2, R2

    MOVD    R1, R6
    ADD     R2, R6, R7
    ADD     R2, R7, R8
    ADD     R2, R8, R9

    VEOR    V0.B16, V0.B16, V0.B16
    VEOR    V1.B16, V1.B16, V1.B16
    VEOR    V2.B16, V2.B16, V2.B16
    VEOR    V3.B16, V3.B16, V3.B16
    VEOR    V4.B16, V4.B16, V4.B16
    VEOR    V5.B16, V5.B16, V5.B16
    VEOR    V6.B16, V6.B16, V6.B16
    VEOR    V7.B16, V7.B16, V7.B16
    VEOR    V8.B16, V8.B16, V8.B16
    VEOR    V9.B16, V9.B16, V9.B16
    VEOR    V10.B16, V10.B16, V10.B16
    VEOR    V11.B16, V11.B16, V11.B16
    VEOR    V12.B16, V12.B16, V12.B16
    VEOR    V13.B16, V13.B16, V13.B16
    VEOR    V14.B16, V14.B16, V14.B16
    VEOR    V15.B16, V15.B16, V15.B16

    CBZ     R0, store

kloop:
    VLD1.P  64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]

    FMOVS   (R6), F20
    VDUP    V20.S[0], V20.S4
    VFMLA   V20.S4, V16.S4, V0.S4
    VFMLA   V20.S4, V17.S4, V1.S4
    VFMLA   V20.S4, V18.S4, V2.S4
    VFMLA   V20.S4, V19.S4, V3.S4

    FMOVS   (R7), F20
    VDUP    V20.S[0], V20.S4
    VFMLA   V20.S4, V16.S4, V4.S4
    VFMLA   V20.S4, V17.S4, V5.S4
    VFMLA   V20.S4, V18.S4, V6.S4
    VFMLA   V20.S4, V19.S4, V7.S4

    FMOVS   (R8), F20
    VDUP    V20.S[0], V20.S4
    VFMLA   V20.S4, V16.S4, V8.S4
    VFMLA   V20.S4, V17.S4, V9.S4
    VFMLA   V20.S4, V18.S4, V10.S4
    VFMLA   V20.S4, V19.S4, V11.S4

    FMOVS   (R9), F20
    VDUP    V20.S[0], V20.S4
    VFMLA   V20.S4, V16.S4, V12.S4
    VFMLA   V20.S4, V17.S4, V13.S4
    VFMLA   V20.S4, V18.S4, V14.S4
    VFMLA   V20.S4, V19.S4, V15.S4

    ADD     $4, R6
    ADD     $4, R7
    ADD     $4, R8
    ADD     $4, R9

    SUB     $1, R0, R0
    CBNZ    R0, kloop

store:
    VDUP    V31.S[0], V20.S4

    // Scale all accumulators by alpha
    FMUL_V20(0, 0)
    FMUL_V20(1, 1)
    FMUL_V20(2, 2)
    FMUL_V20(3, 3)
    FMUL_V20(4, 4)
    FMUL_V20(5, 5)
    FMUL_V20(6, 6)
    FMUL_V20(7, 7)
    FMUL_V20(8, 8)
    FMUL_V20(9, 9)
    FMUL_V20(10, 10)
    FMUL_V20(11, 11)
    FMUL_V20(12, 12)
    FMUL_V20(13, 13)
    FMUL_V20(14, 14)
    FMUL_V20(15, 15)

    MOVD    0(RSP), R4
    MOVD    8(RSP), R5

    // Row 0
    VLD1    (R4), [V16.S4, V17.S4, V18.S4, V19.S4]
    VFADD(0, 16, 16)
    VFADD(1, 17, 17)
    VFADD(2, 18, 18)
    VFADD(3, 19, 19)
    VST1    [V16.S4, V17.S4, V18.S4, V19.S4], (R4)

    ADD     R5, R4
    VLD1    (R4), [V16.S4, V17.S4, V18.S4, V19.S4]
    VFADD(4, 16, 16)
    VFADD(5, 17, 17)
    VFADD(6, 18, 18)
    VFADD(7, 19, 19)
    VST1    [V16.S4, V17.S4, V18.S4, V19.S4], (R4)

    ADD     R5, R4
    VLD1    (R4), [V16.S4, V17.S4, V18.S4, V19.S4]
    VFADD(8, 16, 16)
    VFADD(9, 17, 17)
    VFADD(10, 18, 18)
    VFADD(11, 19, 19)
    VST1    [V16.S4, V17.S4, V18.S4, V19.S4], (R4)

    ADD     R5, R4
    VLD1    (R4), [V16.S4, V17.S4, V18.S4, V19.S4]
    VFADD(12, 16, 16)
    VFADD(13, 17, 17)
    VFADD(14, 18, 18)
    VFADD(15, 19, 19)
    VST1    [V16.S4, V17.S4, V18.S4, V19.S4], (R4)

    RET
