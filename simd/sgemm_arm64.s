// sgemm_arm64.s — ARM64 NEON SGEMM kernels for Go (plan9 assembly)
//
// func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)
// func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)
//
// NEON: 128-bit vectors = 4 × float32.
// Uses VFMLA (fused multiply-add) and VLD1/VST1.
// VFADD is not in Go's arm64 assembler, so we use WORD macros.

#include "textflag.h"

// VFADD Vm.4S, Vn.4S, Vd.4S — manual encoding
// Encoding: 0x4e20d400 | (Vm << 16) | (Vn << 5) | Vd
// VFADD V1.S4, V0.S4, V0.S4 = 0x4e21d400
#define VFADD_V1_V0_V0 WORD $0x4e21d400
// VFADD V3.S4, V2.S4, V2.S4 = 0x4e23d442
#define VFADD_V3_V2_V2 WORD $0x4e23d442
// VFADD V2.S4, V0.S4, V0.S4 = 0x4e22d400
#define VFADD_V2_V0_V0 WORD $0x4e22d400

// Arguments (ABI0 stack layout, 80 bytes):
//   m:     +0(FP)   int
//   n:     +8(FP)   int
//   k:     +16(FP)  int
//   alpha: +24(FP)  float32 (4 bytes + 4 pad)
//   a:     +32(FP)  pointer
//   b:     +40(FP)  pointer
//   c:     +48(FP)  pointer
//   lda:   +56(FP)  int
//   ldb:   +64(FP)  int
//   ldc:   +72(FP)  int

// ============================================================================
// SgemmNT: C += alpha * A * B^T
// ============================================================================
TEXT ·SgemmNT(SB), NOSPLIT, $0-80
    MOVD    m+0(FP), R0           // R0 = m
    MOVD    n+8(FP), R1           // R1 = n
    MOVD    k+16(FP), R2          // R2 = k
    FMOVS   alpha+24(FP), F31     // F31 = alpha
    MOVD    a+32(FP), R3          // R3 = A base
    MOVD    b+40(FP), R4          // R4 = B base
    MOVD    c+48(FP), R5          // R5 = C base
    MOVD    lda+56(FP), R6        // R6 = lda
    MOVD    ldb+64(FP), R7        // R7 = ldb
    MOVD    ldc+72(FP), R8        // R8 = ldc

    // Byte strides
    LSL     $2, R6, R6
    LSL     $2, R7, R7
    LSL     $2, R8, R8

    CBZ     R0, nt_done

    // R9 = A row ptr (current i), R10 = C row ptr (current i)
    MOVD    R3, R9
    MOVD    R5, R10

nt_i_loop:
    MOVD    R1, R11               // R11 = j counter
    MOVD    R4, R12               // R12 = B row ptr
    MOVD    R10, R13              // R13 = &C[i,j]

    CBZ     R11, nt_next_i

nt_j_loop:
    // Dot product: A[i,:] · B[j,:], length k
    MOVD    R9, R14               // R14 = A ptr
    MOVD    R12, R15              // R15 = B ptr
    MOVD    R2, R16               // R16 = k counter

    // Zero accumulators V0-V3
    VEOR    V0.B16, V0.B16, V0.B16
    VEOR    V1.B16, V1.B16, V1.B16
    VEOR    V2.B16, V2.B16, V2.B16
    VEOR    V3.B16, V3.B16, V3.B16

    // Main loop: 16 floats per iteration (4 × 4-wide NEON)
    CMP     $16, R16
    BLT     nt_k8

nt_k16:
    VLD1.P  64(R14), [V4.S4, V5.S4, V6.S4, V7.S4]
    VLD1.P  64(R15), [V16.S4, V17.S4, V18.S4, V19.S4]
    VFMLA   V4.S4, V16.S4, V0.S4
    VFMLA   V5.S4, V17.S4, V1.S4
    VFMLA   V6.S4, V18.S4, V2.S4
    VFMLA   V7.S4, V19.S4, V3.S4
    SUB     $16, R16, R16
    CMP     $16, R16
    BGE     nt_k16

nt_k8:
    // Merge V2,V3 into V0,V1
    VFADD_V3_V2_V2
    VFADD_V2_V0_V0

    CMP     $8, R16
    BLT     nt_k4
    VLD1.P  32(R14), [V4.S4, V5.S4]
    VLD1.P  32(R15), [V16.S4, V17.S4]
    VFMLA   V4.S4, V16.S4, V0.S4
    VFMLA   V5.S4, V17.S4, V1.S4
    SUB     $8, R16, R16

nt_k4:
    VFADD_V1_V0_V0

    CMP     $4, R16
    BLT     nt_k_reduce
    VLD1.P  16(R14), [V4.S4]
    VLD1.P  16(R15), [V16.S4]
    VFMLA   V4.S4, V16.S4, V0.S4
    SUB     $4, R16, R16

nt_k_reduce:
    // Horizontal sum V0.S4 → scalar F20
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

    // Scalar tail
    CBZ     R16, nt_k_store

nt_k_scalar:
    FMOVS   (R14), F4
    FMOVS   (R15), F5
    FMADDS  F4, F0, F5, F0
    ADD     $4, R14
    ADD     $4, R15
    SUB     $1, R16, R16
    CBNZ    R16, nt_k_scalar

nt_k_store:
    // C[i,j] += alpha * dot
    FMULS   F31, F0, F0
    FMOVS   (R13), F4
    FADDS   F0, F4, F4
    FMOVS   F4, (R13)

    ADD     $4, R13              // next C column
    ADD     R7, R12              // next B row
    SUB     $1, R11, R11
    CBNZ    R11, nt_j_loop

nt_next_i:
    ADD     R6, R9               // next A row
    ADD     R8, R10              // next C row
    SUB     $1, R0, R0
    CBNZ    R0, nt_i_loop

nt_done:
    RET


// ============================================================================
// SgemmNN: C += alpha * A * B
// ============================================================================
// Tiles C in 16-element chunks (4 × V128). Keeps accumulators in
// registers across the full k loop.
TEXT ·SgemmNN(SB), NOSPLIT, $0-80
    MOVD    m+0(FP), R0
    MOVD    n+8(FP), R1
    MOVD    k+16(FP), R2
    FMOVS   alpha+24(FP), F31
    MOVD    a+32(FP), R3
    MOVD    b+40(FP), R4
    MOVD    c+48(FP), R5
    MOVD    lda+56(FP), R6
    MOVD    ldb+64(FP), R7
    MOVD    ldc+72(FP), R8

    LSL     $2, R6, R6
    LSL     $2, R7, R7
    LSL     $2, R8, R8

    CBZ     R0, nn_done

    MOVD    R3, R9               // A row ptr
    MOVD    R5, R10              // C row ptr

nn_i_loop:
    MOVD    R1, R11              // remaining columns
    MOVD    R10, R13             // &C[i, jj]
    MOVD    R4, R12              // &B[0, jj]

nn_tile16:
    CMP     $16, R11
    BLT     nn_tile4

    // Load C[i, jj:jj+16]
    VLD1    (R13), [V0.S4, V1.S4, V2.S4, V3.S4]

    MOVD    R9, R14              // &A[i, 0]
    MOVD    R12, R15             // &B[0, jj]
    MOVD    R2, R16              // k counter

nn_p16:
    // Broadcast alpha * A[i, p]
    FMOVS   (R14), F8
    FMULS   F31, F8, F8
    VDUP    V8.S[0], V8.S4

    // C tile += broadcast * B[p, jj:jj+16]
    VLD1    (R15), [V4.S4, V5.S4, V6.S4, V7.S4]
    VFMLA   V8.S4, V4.S4, V0.S4
    VFMLA   V8.S4, V5.S4, V1.S4
    VFMLA   V8.S4, V6.S4, V2.S4
    VFMLA   V8.S4, V7.S4, V3.S4

    ADD     $4, R14
    ADD     R7, R15
    SUB     $1, R16, R16
    CBNZ    R16, nn_p16

    // Store C tile
    VST1    [V0.S4, V1.S4, V2.S4, V3.S4], (R13)

    ADD     $64, R13
    ADD     $64, R12
    SUB     $16, R11, R11
    B       nn_tile16

nn_tile4:
    CMP     $4, R11
    BLT     nn_tile1

    VLD1    (R13), [V0.S4]

    MOVD    R9, R14
    MOVD    R12, R15
    MOVD    R2, R16

nn_p4:
    FMOVS   (R14), F8
    FMULS   F31, F8, F8
    VDUP    V8.S[0], V8.S4
    VLD1    (R15), [V4.S4]
    VFMLA   V8.S4, V4.S4, V0.S4
    ADD     $4, R14
    ADD     R7, R15
    SUB     $1, R16, R16
    CBNZ    R16, nn_p4

    VST1    [V0.S4], (R13)

    ADD     $16, R13
    ADD     $16, R12
    SUB     $4, R11, R11
    B       nn_tile4

nn_tile1:
    CBZ     R11, nn_next_i

nn_tile1_loop:
    FMOVS   (R13), F0

    MOVD    R9, R14
    MOVD    R12, R15
    MOVD    R2, R16

nn_p1:
    FMOVS   (R14), F4
    FMOVS   (R15), F5
    FMULS   F31, F4, F4
    FMADDS  F4, F0, F5, F0
    ADD     $4, R14
    ADD     R7, R15
    SUB     $1, R16, R16
    CBNZ    R16, nn_p1

    FMOVS   F0, (R13)
    ADD     $4, R13
    ADD     $4, R12
    SUB     $1, R11, R11
    CBNZ    R11, nn_tile1_loop

nn_next_i:
    ADD     R6, R9
    ADD     R8, R10
    SUB     $1, R0, R0
    CBNZ    R0, nn_i_loop

nn_done:
    RET
