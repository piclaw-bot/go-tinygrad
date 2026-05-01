// sgemm_blocked_arm64.s — ARM64 NEON blocked NT tile kernel (fixed)

#include "textflag.h"

#define VFADD(vm, vn, vd) WORD $(0x4e20d400 | ((vm)<<16) | ((vn)<<5) | (vd))

TEXT ·sgemmNTTileFMA(SB), NOSPLIT, $64-80
    MOVD    iLen+0(FP), R19        // use R19 for i counter (callee-saved in Go ABI)
    MOVD    jLen+8(FP), R20        // R20 = jLen (constant)
    MOVD    kLen+16(FP), R21       // R21 = kLen (constant)
    FMOVS   alpha+24(FP), F31
    MOVD    a+32(FP), R22          // R22 = A row ptr (advances per i)
    MOVD    lda+40(FP), R23        // R23 = lda
    MOVD    b+48(FP), R24          // R24 = B base (constant)
    MOVD    ldb+56(FP), R25        // R25 = ldb
    MOVD    c+64(FP), R26          // R26 = C row ptr (advances per i)
    MOVD    ldc+72(FP), R27        // R27 = ldc

    LSL     $2, R23, R23           // lda_bytes
    LSL     $2, R25, R25           // ldb_bytes
    LSL     $2, R27, R27           // ldc_bytes

    CBZ     R19, done

i_loop:
    MOVD    R20, R11               // j counter
    MOVD    R24, R12               // B row ptr
    MOVD    R26, R13               // C element ptr

j_loop:
    CBZ     R11, next_i

    // k-loop: dot(A[i, 0:kLen], B[j, 0:kLen])
    MOVD    R22, R14               // A ptr (from current A row)
    MOVD    R12, R15               // B ptr (from current B row)
    MOVD    R21, R16               // k counter

    VEOR    V0.B16, V0.B16, V0.B16
    VEOR    V1.B16, V1.B16, V1.B16

    CMP     $16, R16
    BLT     k_scalar_check

k4_loop:
    VLD1.P  64(R14), [V4.S4, V5.S4, V6.S4, V7.S4]
    VLD1.P  64(R15), [V16.S4, V17.S4, V18.S4, V19.S4]
    VFMLA   V4.S4, V16.S4, V0.S4
    VFMLA   V5.S4, V17.S4, V1.S4
    VFMLA   V6.S4, V18.S4, V0.S4
    VFMLA   V7.S4, V19.S4, V1.S4
    SUB     $16, R16, R16
    CMP     $16, R16
    BGE     k4_loop

    // Merge V1 into V0
    VFADD(1, 0, 0)

k_scalar_check:
    // Horizontal sum V0
    VMOV    V0.S[0], R17
    FMOVS   R17, F20
    VMOV    V0.S[1], R17
    FMOVS   R17, F4
    FADDS   F4, F20, F20
    VMOV    V0.S[2], R17
    FMOVS   R17, F4
    FADDS   F4, F20, F20
    VMOV    V0.S[3], R17
    FMOVS   R17, F4
    FADDS   F4, F20, F20

    CBZ     R16, k_store

k_scalar:
    FMOVS   (R14), F4
    FMOVS   (R15), F5
    FMADDS  F4, F20, F5, F20        // F0 = F5*F4 + F0
    ADD     $4, R14
    ADD     $4, R15
    SUB     $1, R16, R16
    CBNZ    R16, k_scalar

k_store:
    // C[i,j] += alpha * dot
    FMULS   F31, F20, F20
    FMOVS   (R13), F4
    FADDS   F20, F4, F4
    FMOVS   F4, (R13)

    // Advance j
    ADD     $4, R13                // next C column
    ADD     R25, R12               // next B row
    SUB     $1, R11, R11
    B       j_loop

next_i:
    ADD     R23, R22               // A next row
    ADD     R27, R26               // C next row
    SUB     $1, R19, R19
    CBNZ    R19, i_loop

done:
    RET
