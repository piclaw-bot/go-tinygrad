// simd_amd64.s — AVX2/FMA SIMD kernels for Go (plan9 assembly)

#include "textflag.h"

// func Sdot(x, y []float32) float32
TEXT ·Sdot(SB), NOSPLIT, $0-52
    MOVQ    x_base+0(FP), SI
    MOVQ    x_len+8(FP), CX
    MOVQ    y_base+24(FP), DI

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    CMPQ    CX, $16
    JL      sdot_post16

sdot_loop16:
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS (DI), Y2, Y0
    VFMADD231PS 32(DI), Y3, Y1
    ADDQ    $64, SI
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     sdot_loop16

sdot_post16:
    VADDPS  Y1, Y0, Y0

    CMPQ    CX, $8
    JL      sdot_post8
    VMOVUPS (SI), Y2
    VFMADD231PS (DI), Y2, Y0
    ADDQ    $32, SI
    ADDQ    $32, DI
    SUBQ    $8, CX

sdot_post8:
    // Horizontal reduce Y0 → scalar in X0
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    // 4-wide tail (into X4, then add to X0)
    CMPQ    CX, $4
    JL      sdot_scalar_check
    VXORPS  X4, X4, X4
    VMOVUPS (SI), X2
    VMOVUPS (DI), X3
    VFMADD231PS X3, X2, X4
    // hsum X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4
    VADDSS  X4, X0, X0
    ADDQ    $16, SI
    ADDQ    $16, DI
    SUBQ    $4, CX

sdot_scalar_check:
    TESTQ   CX, CX
    JZ      sdot_done

sdot_scalar:
    VMOVSS  (SI), X1
    VMOVSS  (DI), X2
    VFMADD231SS X2, X1, X0
    ADDQ    $4, SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     sdot_scalar

sdot_done:
    VMOVSS  X0, ret+48(FP)
    VZEROUPPER
    RET

// func Saxpy(alpha float32, x []float32, y []float32)
TEXT ·Saxpy(SB), NOSPLIT, $0-56
    MOVSS       alpha+0(FP), X8
    VBROADCASTSS X8, Y8
    MOVQ    x_base+8(FP), SI
    MOVQ    x_len+16(FP), CX
    MOVQ    y_base+32(FP), DI

    CMPQ    CX, $16
    JL      saxpy_post16

saxpy_loop16:
    VMOVUPS (DI), Y0
    VMOVUPS 32(DI), Y1
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS Y8, Y2, Y0
    VFMADD231PS Y8, Y3, Y1
    VMOVUPS Y0, (DI)
    VMOVUPS Y1, 32(DI)
    ADDQ    $64, SI
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     saxpy_loop16

saxpy_post16:
    CMPQ    CX, $8
    JL      saxpy_post8
    VMOVUPS (DI), Y0
    VMOVUPS (SI), Y2
    VFMADD231PS Y8, Y2, Y0
    VMOVUPS Y0, (DI)
    ADDQ    $32, SI
    ADDQ    $32, DI
    SUBQ    $8, CX

saxpy_post8:
    CMPQ    CX, $4
    JL      saxpy_scalar_check
    VMOVUPS (DI), X0
    VMOVUPS (SI), X2
    VFMADD231PS X8, X2, X0
    VMOVUPS X0, (DI)
    ADDQ    $16, SI
    ADDQ    $16, DI
    SUBQ    $4, CX

saxpy_scalar_check:
    TESTQ   CX, CX
    JZ      saxpy_done

saxpy_scalar:
    VMOVSS  (DI), X0
    VMOVSS  (SI), X2
    VFMADD231SS X8, X2, X0
    VMOVSS  X0, (DI)
    ADDQ    $4, SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     saxpy_scalar

saxpy_done:
    VZEROUPPER
    RET
