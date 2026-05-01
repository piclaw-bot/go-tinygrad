// sgemm_gather_amd64.s — AVX2 VGATHERDPS NT micro-kernel
//
// func gatherMicroKernel6x8(k int, alpha float32, a unsafe.Pointer, lda int,
//                            b unsafe.Pointer, indices unsafe.Pointer,
//                            c unsafe.Pointer, ldc int)
//
// Computes C[0:6, 0:8] += alpha * A[0:6, 0:k] · B[gathered, 0:k]^T
//
// For each k step:
//   VGATHERDPS: load B[jj+0,p]..B[jj+7,p] using index vector (1 instruction)
//   6× VBROADCASTSS A[row,p] + VFMADD231PS into accumulator
//   = 1 gather + 6 broadcasts + 6 FMA = 13 ops for 48 results
//   NO horizontal reduction
//
// Args (64 bytes):
//   k:0  alpha:8(+4pad)  a:16  lda:24  b:32  indices:40  c:48  ldc:56

#include "textflag.h"

TEXT ·gatherMicroKernel6x8(SB), NOSPLIT, $16-64
    MOVQ    k+0(FP), CX
    MOVQ    a+16(FP), AX
    MOVQ    lda+24(FP), DX
    MOVQ    b+32(FP), SI            // B[jj, 0]
    MOVQ    indices+40(FP), BX
    MOVQ    c+48(FP), DI
    MOVQ    ldc+56(FP), R15

    // Save C and ldc_bytes for store phase
    MOVQ    DI, 0(SP)
    SHLQ    $2, R15
    MOVQ    R15, 8(SP)

    // Load gather index vector (constant across all k steps)
    VMOVDQU (BX), Y8                // Y8 = [0, ldb, 2*ldb, ..., 7*ldb]

    // Compute 6 A row pointers
    SHLQ    $2, DX                  // lda_bytes
    MOVQ    AX, R8                  // row 0
    LEAQ    (AX)(DX*1), R9          // row 1
    LEAQ    (R9)(DX*1), R10         // row 2
    LEAQ    (R10)(DX*1), R11        // row 3
    LEAQ    (R11)(DX*1), R12        // row 4
    LEAQ    (R12)(DX*1), R13        // row 5

    // Zero 6 accumulators (C tile: 6 rows × 8 cols)
    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1
    VXORPS  Y2, Y2, Y2
    VXORPS  Y3, Y3, Y3
    VXORPS  Y4, Y4, Y4
    VXORPS  Y5, Y5, Y5

    TESTQ   CX, CX
    JZ      store

kloop:
    // Gather 8 B values: B[jj+d, p] for d=0..7
    // VGATHERDPS uses: dest[i] = mem[base + index[i]*scale]
    // base = SI (pointer to B[jj, p]), index = Y8 (row offsets), scale = 4 (float32)
    // Mask must be all-ones; VGATHERDPS zeroes mask bits as it loads.
    VPCMPEQD Y9, Y9, Y9            // Y9 = all-ones mask
    VGATHERDPS Y9, (SI)(Y8*4), Y6  // Y6 = [B[jj,p], B[jj+1,p], ..., B[jj+7,p]]

    // Row 0: broadcast A[0,p], FMA
    VBROADCASTSS (R8), Y7
    VFMADD231PS Y7, Y6, Y0

    // Row 1
    VBROADCASTSS (R9), Y7
    VFMADD231PS Y7, Y6, Y1

    // Row 2
    VBROADCASTSS (R10), Y7
    VFMADD231PS Y7, Y6, Y2

    // Row 3
    VBROADCASTSS (R11), Y7
    VFMADD231PS Y7, Y6, Y3

    // Row 4
    VBROADCASTSS (R12), Y7
    VFMADD231PS Y7, Y6, Y4

    // Row 5
    VBROADCASTSS (R13), Y7
    VFMADD231PS Y7, Y6, Y5

    // Advance: A rows by 4, B by 4 (next column p+1)
    ADDQ    $4, R8
    ADDQ    $4, R9
    ADDQ    $4, R10
    ADDQ    $4, R11
    ADDQ    $4, R12
    ADDQ    $4, R13
    ADDQ    $4, SI

    DECQ    CX
    JNZ     kloop

store:
    // Scale by alpha and add to C
    VBROADCASTSS alpha+8(FP), Y7
    MOVQ    0(SP), DI               // C base
    MOVQ    8(SP), R15              // ldc_bytes

    VMULPS  Y7, Y0, Y0
    VADDPS  (DI), Y0, Y0
    VMOVUPS Y0, (DI)

    ADDQ    R15, DI
    VMULPS  Y7, Y1, Y1
    VADDPS  (DI), Y1, Y1
    VMOVUPS Y1, (DI)

    ADDQ    R15, DI
    VMULPS  Y7, Y2, Y2
    VADDPS  (DI), Y2, Y2
    VMOVUPS Y2, (DI)

    ADDQ    R15, DI
    VMULPS  Y7, Y3, Y3
    VADDPS  (DI), Y3, Y3
    VMOVUPS Y3, (DI)

    ADDQ    R15, DI
    VMULPS  Y7, Y4, Y4
    VADDPS  (DI), Y4, Y4
    VMOVUPS Y4, (DI)

    ADDQ    R15, DI
    VMULPS  Y7, Y5, Y5
    VADDPS  (DI), Y5, Y5
    VMOVUPS Y5, (DI)

    VZEROUPPER
    RET
