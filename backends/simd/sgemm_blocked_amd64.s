// sgemm_blocked_amd64.s — AVX2/FMA blocked NT tile kernel
// 2j at a time, sharing A loads. Cache-blocked by Go driver.

#include "textflag.h"

TEXT ·sgemmNTTileFMA(SB), NOSPLIT, $40-80
    MOVQ    iLen+0(FP), R8
    MOVQ    jLen+8(FP), R14
    MOVQ    kLen+16(FP), R13
    VBROADCASTSS alpha+24(FP), Y15
    MOVQ    a+32(FP), SI
    MOVQ    lda+40(FP), R11
    MOVQ    b+48(FP), R12
    MOVQ    ldb+56(FP), DX
    MOVQ    c+64(FP), DI
    MOVQ    ldc+72(FP), R15

    SHLQ    $2, R11
    SHLQ    $2, DX
    MOVQ    DX, 24(SP)
    SHLQ    $2, R15

    TESTQ   R8, R8
    JZ      done

i_loop:
    MOVQ    R14, R9
    MOVQ    R12, R10
    MOVQ    DI, CX

j_loop2:
    CMPQ    R9, $2
    JL      j_loop1

    MOVQ    R9, 0(SP)
    MOVQ    CX, 8(SP)
    MOVQ    R10, 16(SP)
    MOVQ    24(SP), DX
    LEAQ    (R10)(DX*1), AX
    MOVQ    AX, 32(SP)

    MOVQ    SI, AX
    MOVQ    R10, BX
    MOVQ    32(SP), DX
    MOVQ    R13, CX

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1
    VXORPS  Y4, Y4, Y4
    VXORPS  Y5, Y5, Y5

    CMPQ    CX, $16
    JL      k2_8

k2_16:
    VMOVUPS    (AX), Y8
    VMOVUPS  32(AX), Y9
    VFMADD231PS (BX), Y8, Y0
    VFMADD231PS 32(BX), Y9, Y1
    VFMADD231PS (DX), Y8, Y4
    VFMADD231PS 32(DX), Y9, Y5
    ADDQ    $64, AX
    ADDQ    $64, BX
    ADDQ    $64, DX
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     k2_16

k2_8:
    VADDPS  Y1, Y0, Y0
    VADDPS  Y5, Y4, Y4
    CMPQ    CX, $8
    JL      k2_reduce
    VMOVUPS    (AX), Y8
    VFMADD231PS (BX), Y8, Y0
    VFMADD231PS (DX), Y8, Y4
    ADDQ    $32, AX
    ADDQ    $32, BX
    ADDQ    $32, DX
    SUBQ    $8, CX

k2_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VEXTRACTF128 $1, Y4, X6
    VADDPS  X6, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4

    TESTQ   CX, CX
    JZ      k2_store

k2_scalar:
    VMOVSS  (AX), X8
    VMOVSS  (BX), X9
    VFMADD231SS X9, X8, X0
    VMOVSS  (DX), X9
    VFMADD231SS X9, X8, X4
    ADDQ    $4, AX
    ADDQ    $4, BX
    ADDQ    $4, DX
    DECQ    CX
    JNZ     k2_scalar

k2_store:
    MOVQ    0(SP), R9
    MOVQ    8(SP), CX
    MOVQ    16(SP), R10

    VMULSS  X15, X0, X0
    VADDSS  (CX), X0, X0
    VMOVSS  X0, (CX)
    VMULSS  X15, X4, X4
    VADDSS  4(CX), X4, X4
    VMOVSS  X4, 4(CX)

    ADDQ    $8, CX
    MOVQ    24(SP), DX
    LEAQ    (R10)(DX*2), R10
    SUBQ    $2, R9
    JMP     j_loop2

j_loop1:
    TESTQ   R9, R9
    JZ      next_i

    MOVQ    SI, AX
    MOVQ    R10, BX
    MOVQ    R13, R9

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    CMPQ    R9, $16
    JL      k1_8

k1_16:
    VMOVUPS    (AX), Y8
    VMOVUPS  32(AX), Y9
    VFMADD231PS (BX), Y8, Y0
    VFMADD231PS 32(BX), Y9, Y1
    ADDQ    $64, AX
    ADDQ    $64, BX
    SUBQ    $16, R9
    CMPQ    R9, $16
    JGE     k1_16

k1_8:
    VADDPS  Y1, Y0, Y0
    CMPQ    R9, $8
    JL      k1_reduce
    VMOVUPS (AX), Y8
    VFMADD231PS (BX), Y8, Y0
    ADDQ    $32, AX
    ADDQ    $32, BX
    SUBQ    $8, R9

k1_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    TESTQ   R9, R9
    JZ      k1_store

k1_scalar:
    VMOVSS  (AX), X8
    VMOVSS  (BX), X9
    VFMADD231SS X9, X8, X0
    ADDQ    $4, AX
    ADDQ    $4, BX
    DECQ    R9
    JNZ     k1_scalar

k1_store:
    VMULSS  X15, X0, X0
    VADDSS  (CX), X0, X0
    VMOVSS  X0, (CX)

next_i:
    ADDQ    R11, SI
    ADDQ    R15, DI
    DECQ    R8
    JNZ     i_loop

done:
    VZEROUPPER
    RET
