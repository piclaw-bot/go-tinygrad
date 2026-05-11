// sgemm_amd64.s — AVX2/FMA SGEMM kernels for Go (plan9 assembly)
//
// func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)
// func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)
//
// NT: C += alpha * A * B^T.  Inner k-loop: 4-way unrolled FMA dot product.
// NN: C += alpha * A * B.    Tiles C in 32-element chunks, k-loop inside.
// Both assume caller has pre-scaled C by beta.

#include "textflag.h"

// ============================================================================
// SgemmNT
// ============================================================================
// Stack frame: 16 bytes local storage
//   0(SP)  = saved j counter (CX)
//   8(SP)  = saved C[i,j] ptr (R9)
TEXT ·SgemmNT(SB), NOSPLIT, $16-80
    MOVQ    m+0(FP), R8
    MOVQ    n+8(FP), DX
    MOVQ    k+16(FP), R10
    MOVQ    a+32(FP), SI
    MOVQ    b+40(FP), R12
    MOVQ    c+48(FP), DI
    MOVQ    lda+56(FP), R11
    MOVQ    ldb+64(FP), R13
    MOVQ    ldc+72(FP), R15

    VBROADCASTSS alpha+24(FP), Y15

    SHLQ    $2, R11
    SHLQ    $2, R13
    SHLQ    $2, R15

    TESTQ   R8, R8
    JZ      nt_done

nt_i_loop:
    MOVQ    DX, CX
    MOVQ    R12, R14
    MOVQ    DI, R9

    TESTQ   CX, CX
    JZ      nt_next_i

nt_j_loop:
    // Save j state
    MOVQ    CX, 0(SP)
    MOVQ    R9, 8(SP)

    // k-loop: dot(A[i,:], B[j,:])
    MOVQ    SI, AX
    MOVQ    R14, BX
    MOVQ    R10, CX

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1
    VXORPS  Y2, Y2, Y2
    VXORPS  Y3, Y3, Y3

    CMPQ    CX, $32
    JL      nt_k16

nt_k32:
    VMOVUPS    (AX), Y4
    VFMADD231PS (BX), Y4, Y0
    VMOVUPS  32(AX), Y5
    VFMADD231PS 32(BX), Y5, Y1
    VMOVUPS  64(AX), Y6
    VFMADD231PS 64(BX), Y6, Y2
    VMOVUPS  96(AX), Y7
    VFMADD231PS 96(BX), Y7, Y3
    ADDQ    $128, AX
    ADDQ    $128, BX
    SUBQ    $32, CX
    CMPQ    CX, $32
    JGE     nt_k32

nt_k16:
    VADDPS  Y2, Y0, Y0
    VADDPS  Y3, Y1, Y1

    CMPQ    CX, $16
    JL      nt_k8
    VMOVUPS    (AX), Y4
    VFMADD231PS (BX), Y4, Y0
    VMOVUPS  32(AX), Y5
    VFMADD231PS 32(BX), Y5, Y1
    ADDQ    $64, AX
    ADDQ    $64, BX
    SUBQ    $16, CX

nt_k8:
    VADDPS  Y1, Y0, Y0

    CMPQ    CX, $8
    JL      nt_k_reduce
    VMOVUPS    (AX), Y4
    VFMADD231PS (BX), Y4, Y0
    ADDQ    $32, AX
    ADDQ    $32, BX
    SUBQ    $8, CX

nt_k_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    TESTQ   CX, CX
    JZ      nt_k_store

nt_k_scalar:
    VMOVSS  (AX), X4
    VMOVSS  (BX), X5
    VFMADD231SS X5, X4, X0
    ADDQ    $4, AX
    ADDQ    $4, BX
    DECQ    CX
    JNZ     nt_k_scalar

nt_k_store:
    // Restore j state
    MOVQ    0(SP), CX
    MOVQ    8(SP), R9

    // C[i,j] += alpha * dot
    VMULSS  X15, X0, X0
    VADDSS  (R9), X0, X0
    VMOVSS  X0, (R9)

    ADDQ    $4, R9
    ADDQ    R13, R14
    DECQ    CX
    JNZ     nt_j_loop

nt_next_i:
    ADDQ    R11, SI
    ADDQ    R15, DI
    DECQ    R8
    JNZ     nt_i_loop

nt_done:
    VZEROUPPER
    RET


// ============================================================================
// SgemmNN
// ============================================================================
// Stack frame: 8 bytes (saved j counter)
TEXT ·SgemmNN(SB), NOSPLIT, $8-80
    MOVQ    m+0(FP), R8
    MOVQ    n+8(FP), DX
    MOVQ    k+16(FP), R10
    MOVQ    a+32(FP), SI
    MOVQ    b+40(FP), R12
    MOVQ    c+48(FP), DI
    MOVQ    lda+56(FP), R11
    MOVQ    ldb+64(FP), R13
    MOVQ    ldc+72(FP), R15

    VBROADCASTSS alpha+24(FP), Y15

    SHLQ    $2, R11
    SHLQ    $2, R13
    SHLQ    $2, R15

    TESTQ   R8, R8
    JZ      nn_done

nn_i_loop:
    MOVQ    DX, CX
    MOVQ    DI, R9
    MOVQ    R12, R14

nn_tile32:
    CMPQ    CX, $32
    JL      nn_tile8

    // Load C[i, jj:jj+32]
    VMOVUPS    (R9), Y0
    VMOVUPS  32(R9), Y1
    VMOVUPS  64(R9), Y2
    VMOVUPS  96(R9), Y3

    MOVQ    SI, AX
    MOVQ    R14, BX
    MOVQ    CX, 0(SP)
    MOVQ    R10, CX

nn_p32:
    VMOVSS     (AX), X8
    VMULSS  X15, X8, X8
    VBROADCASTSS X8, Y8
    VFMADD231PS    (BX), Y8, Y0
    VFMADD231PS  32(BX), Y8, Y1
    VFMADD231PS  64(BX), Y8, Y2
    VFMADD231PS  96(BX), Y8, Y3
    ADDQ    $4, AX
    ADDQ    R13, BX
    DECQ    CX
    JNZ     nn_p32

    MOVQ    0(SP), CX
    VMOVUPS  Y0,    (R9)
    VMOVUPS  Y1,  32(R9)
    VMOVUPS  Y2,  64(R9)
    VMOVUPS  Y3,  96(R9)

    ADDQ    $128, R9
    ADDQ    $128, R14
    SUBQ    $32, CX
    JMP     nn_tile32

nn_tile8:
    CMPQ    CX, $8
    JL      nn_tile1

    VMOVUPS (R9), Y0
    MOVQ    SI, AX
    MOVQ    R14, BX
    MOVQ    CX, 0(SP)
    MOVQ    R10, CX

nn_p8:
    VMOVSS     (AX), X8
    VMULSS  X15, X8, X8
    VBROADCASTSS X8, Y8
    VFMADD231PS (BX), Y8, Y0
    ADDQ    $4, AX
    ADDQ    R13, BX
    DECQ    CX
    JNZ     nn_p8

    MOVQ    0(SP), CX
    VMOVUPS Y0, (R9)

    ADDQ    $32, R9
    ADDQ    $32, R14
    SUBQ    $8, CX
    JMP     nn_tile8

nn_tile1:
    TESTQ   CX, CX
    JZ      nn_next_i

nn_tile1_loop:
    VMOVSS  (R9), X0
    MOVQ    SI, AX
    MOVQ    R14, BX
    MOVQ    CX, 0(SP)
    MOVQ    R10, CX

nn_p1:
    VMOVSS  (AX), X4
    VMOVSS  (BX), X5
    VMULSS  X15, X4, X4
    VFMADD231SS X5, X4, X0
    ADDQ    $4, AX
    ADDQ    R13, BX
    DECQ    CX
    JNZ     nn_p1

    MOVQ    0(SP), CX
    VMOVSS  X0, (R9)

    ADDQ    $4, R9
    ADDQ    $4, R14
    DECQ    CX
    JNZ     nn_tile1_loop

nn_next_i:
    ADDQ    R11, SI
    ADDQ    R15, DI
    DECQ    R8
    JNZ     nn_i_loop

nn_done:
    VZEROUPPER
    RET
