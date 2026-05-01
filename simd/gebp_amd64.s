#include "textflag.h"

// gebpMicroKernel: 6×16 GEBP tile, AVX2+FMA
// C[0:6, 0:16] += alpha * A[0:6, 0:k] × Bp[0:k, 0:16]
//
// Args (56 bytes):
//   k:0  alpha:8  a:16  lda:24  bp:32  c:40  ldc:48

TEXT ·gebpMicroKernel(SB), NOSPLIT, $16-56
    MOVQ    k+0(FP), CX
    MOVQ    a+16(FP), AX
    MOVQ    lda+24(FP), DX
    MOVQ    bp+32(FP), SI
    MOVQ    c+40(FP), DI
    MOVQ    ldc+48(FP), BX

    MOVQ    DI, 0(SP)
    SHLQ    $2, BX
    MOVQ    BX, 8(SP)
    SHLQ    $2, DX

    MOVQ    AX, R8
    LEAQ    (AX)(DX*1), R9
    LEAQ    (R9)(DX*1), R10
    LEAQ    (R10)(DX*1), R11
    LEAQ    (R11)(DX*1), R12
    LEAQ    (R12)(DX*1), R13

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1
    VXORPS  Y2, Y2, Y2
    VXORPS  Y3, Y3, Y3
    VXORPS  Y4, Y4, Y4
    VXORPS  Y5, Y5, Y5
    VXORPS  Y6, Y6, Y6
    VXORPS  Y7, Y7, Y7
    VXORPS  Y8, Y8, Y8
    VXORPS  Y9, Y9, Y9
    VXORPS  Y10, Y10, Y10
    VXORPS  Y11, Y11, Y11

    TESTQ   CX, CX
    JZ      store

kloop:
    VMOVUPS    (SI), Y12
    VMOVUPS  32(SI), Y13

    VBROADCASTSS (R8), Y14
    VFMADD231PS  Y14, Y12, Y0
    VFMADD231PS  Y14, Y13, Y1

    VBROADCASTSS (R9), Y14
    VFMADD231PS  Y14, Y12, Y2
    VFMADD231PS  Y14, Y13, Y3

    VBROADCASTSS (R10), Y14
    VFMADD231PS  Y14, Y12, Y4
    VFMADD231PS  Y14, Y13, Y5

    VBROADCASTSS (R11), Y14
    VFMADD231PS  Y14, Y12, Y6
    VFMADD231PS  Y14, Y13, Y7

    VBROADCASTSS (R12), Y14
    VFMADD231PS  Y14, Y12, Y8
    VFMADD231PS  Y14, Y13, Y9

    VBROADCASTSS (R13), Y14
    VFMADD231PS  Y14, Y12, Y10
    VFMADD231PS  Y14, Y13, Y11

    ADDQ    $4, R8
    ADDQ    $4, R9
    ADDQ    $4, R10
    ADDQ    $4, R11
    ADDQ    $4, R12
    ADDQ    $4, R13
    ADDQ    $64, SI

    DECQ    CX
    JNZ     kloop

store:
    VBROADCASTSS alpha+8(FP), Y14
    MOVQ    0(SP), DI
    MOVQ    8(SP), BX

    VMULPS  Y14, Y0, Y0
    VMULPS  Y14, Y1, Y1
    VADDPS     (DI), Y0, Y0
    VADDPS   32(DI), Y1, Y1
    VMOVUPS  Y0,    (DI)
    VMOVUPS  Y1,  32(DI)

    ADDQ    BX, DI
    VMULPS  Y14, Y2, Y2
    VMULPS  Y14, Y3, Y3
    VADDPS     (DI), Y2, Y2
    VADDPS   32(DI), Y3, Y3
    VMOVUPS  Y2,    (DI)
    VMOVUPS  Y3,  32(DI)

    ADDQ    BX, DI
    VMULPS  Y14, Y4, Y4
    VMULPS  Y14, Y5, Y5
    VADDPS     (DI), Y4, Y4
    VADDPS   32(DI), Y5, Y5
    VMOVUPS  Y4,    (DI)
    VMOVUPS  Y5,  32(DI)

    ADDQ    BX, DI
    VMULPS  Y14, Y6, Y6
    VMULPS  Y14, Y7, Y7
    VADDPS     (DI), Y6, Y6
    VADDPS   32(DI), Y7, Y7
    VMOVUPS  Y6,    (DI)
    VMOVUPS  Y7,  32(DI)

    ADDQ    BX, DI
    VMULPS  Y14, Y8, Y8
    VMULPS  Y14, Y9, Y9
    VADDPS     (DI), Y8, Y8
    VADDPS   32(DI), Y9, Y9
    VMOVUPS  Y8,    (DI)
    VMOVUPS  Y9,  32(DI)

    ADDQ    BX, DI
    VMULPS  Y14, Y10, Y10
    VMULPS  Y14, Y11, Y11
    VADDPS     (DI), Y10, Y10
    VADDPS   32(DI), Y11, Y11
    VMOVUPS  Y10,    (DI)
    VMOVUPS  Y11,  32(DI)

    VZEROUPPER
    RET
