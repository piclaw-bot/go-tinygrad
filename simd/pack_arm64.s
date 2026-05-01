// pack_arm64.s — NEON 4×4 transpose pack for GEBP
//
// func packBNTAsm(b0..b15 uintptr, k int, bp uintptr)
//
// Processes 4 columns at a time using 4×4 transpose via VTRN1/VTRN2/VZIP1/VZIP2.
// For each group of 4 rows and 4 columns:
//   Load V0=[r0[p:p+4]], V1=[r1[p:p+4]], V2=[r2[p:p+4]], V3=[r3[p:p+4]]
//   Transpose → V0'=[r0p,r1p,r2p,r3p], V1'=[r0p+1,...], etc.
//   Store at bp offsets for 4 consecutive p values.

#include "textflag.h"

TEXT ·packBNTAsm(SB), NOSPLIT, $0-144
    MOVD    b0+0(FP), R0
    MOVD    b1+8(FP), R1
    MOVD    b2+16(FP), R2
    MOVD    b3+24(FP), R3
    MOVD    b4+32(FP), R4
    MOVD    b5+40(FP), R5
    MOVD    b6+48(FP), R6
    MOVD    b7+56(FP), R7
    MOVD    b8+64(FP), R8
    MOVD    b9+72(FP), R9
    MOVD    b10+80(FP), R10
    MOVD    b11+88(FP), R11
    MOVD    b12+96(FP), R12
    MOVD    b13+104(FP), R13
    MOVD    b14+112(FP), R14
    MOVD    b15+120(FP), R15
    MOVD    k+128(FP), R16
    MOVD    bp+136(FP), R17

    // Process 4 columns at a time
    CMP     $4, R16
    BLT     tail1

loop4:
    // Group 0: rows 0-3, load 4 floats each
    VLD1.P  16(R0), [V0.S4]       // r0[p:p+4]
    VLD1.P  16(R1), [V1.S4]       // r1[p:p+4]
    VLD1.P  16(R2), [V2.S4]       // r2[p:p+4]
    VLD1.P  16(R3), [V3.S4]       // r3[p:p+4]
    // 4×4 transpose
    VTRN1   V1.S4, V0.S4, V16.S4  // [r0p0, r1p0, r0p2, r1p2]
    VTRN2   V1.S4, V0.S4, V17.S4  // [r0p1, r1p1, r0p3, r1p3]
    VTRN1   V3.S4, V2.S4, V18.S4  // [r2p0, r3p0, r2p2, r3p2]
    VTRN2   V3.S4, V2.S4, V19.S4  // [r2p1, r3p1, r2p3, r3p3]
    VZIP1   V18.D2, V16.D2, V20.D2 // [r0p0, r1p0, r2p0, r3p0] = col p+0, rows 0-3
    VZIP1   V19.D2, V17.D2, V21.D2 // col p+1, rows 0-3
    VZIP2   V18.D2, V16.D2, V22.D2 // col p+2, rows 0-3
    VZIP2   V19.D2, V17.D2, V23.D2 // col p+3, rows 0-3

    // Group 1: rows 4-7
    VLD1.P  16(R4), [V0.S4]
    VLD1.P  16(R5), [V1.S4]
    VLD1.P  16(R6), [V2.S4]
    VLD1.P  16(R7), [V3.S4]
    VTRN1   V1.S4, V0.S4, V16.S4
    VTRN2   V1.S4, V0.S4, V17.S4
    VTRN1   V3.S4, V2.S4, V18.S4
    VTRN2   V3.S4, V2.S4, V19.S4
    VZIP1   V18.D2, V16.D2, V24.D2 // col p+0, rows 4-7
    VZIP1   V19.D2, V17.D2, V25.D2 // col p+1, rows 4-7
    VZIP2   V18.D2, V16.D2, V26.D2 // col p+2, rows 4-7
    VZIP2   V19.D2, V17.D2, V27.D2 // col p+3, rows 4-7

    // Group 2: rows 8-11
    VLD1.P  16(R8), [V0.S4]
    VLD1.P  16(R9), [V1.S4]
    VLD1.P  16(R10), [V2.S4]
    VLD1.P  16(R11), [V3.S4]
    VTRN1   V1.S4, V0.S4, V16.S4
    VTRN2   V1.S4, V0.S4, V17.S4
    VTRN1   V3.S4, V2.S4, V18.S4
    VTRN2   V3.S4, V2.S4, V19.S4
    VZIP1   V18.D2, V16.D2, V28.D2 // col p+0, rows 8-11
    VZIP1   V19.D2, V17.D2, V29.D2
    VZIP2   V18.D2, V16.D2, V30.D2
    VZIP2   V19.D2, V17.D2, V31.D2

    // Group 3: rows 12-15
    VLD1.P  16(R12), [V0.S4]
    VLD1.P  16(R13), [V1.S4]
    VLD1.P  16(R14), [V2.S4]
    VLD1.P  16(R15), [V3.S4]
    VTRN1   V1.S4, V0.S4, V16.S4
    VTRN2   V1.S4, V0.S4, V17.S4
    VTRN1   V3.S4, V2.S4, V18.S4
    VTRN2   V3.S4, V2.S4, V19.S4
    VZIP1   V18.D2, V16.D2, V0.D2  // col p+0, rows 12-15
    VZIP1   V19.D2, V17.D2, V1.D2
    VZIP2   V18.D2, V16.D2, V2.D2
    VZIP2   V19.D2, V17.D2, V3.D2

    // Store 4 packed columns: each is 16 floats = 64 bytes
    // Column p+0: [V20(rows 0-3), V24(rows 4-7), V28(rows 8-11), V0(rows 12-15)]
    VST1    [V20.S4, V21.S4], (R17)
    ADD     $32, R17
    VST1    [V22.S4, V23.S4], (R17)
    // Wait — this is wrong. Each column of the packed output has 16 floats
    // stored contiguously: bp[p*16 + 0..15].
    // Column p+0: rows 0-3=V20, rows 4-7=V24, rows 8-11=V28, rows 12-15=V0
    // Column p+1: V21, V25, V29, V1
    // etc.
    // Reset R17 and store properly.
    SUB     $32, R17

    // Column p+0 (offset 0): V20, V24, V28, V0
    VST1    [V20.S4], (R17)
    ADD     $16, R17
    VST1    [V24.S4], (R17)
    ADD     $16, R17
    VST1    [V28.S4], (R17)
    ADD     $16, R17
    VST1    [V0.S4], (R17)
    ADD     $16, R17
    // Column p+1 (offset 64): V21, V25, V29, V1
    VST1    [V21.S4], (R17)
    ADD     $16, R17
    VST1    [V25.S4], (R17)
    ADD     $16, R17
    VST1    [V29.S4], (R17)
    ADD     $16, R17
    VST1    [V1.S4], (R17)
    ADD     $16, R17
    // Column p+2 (offset 128): V22, V26, V30, V2
    VST1    [V22.S4], (R17)
    ADD     $16, R17
    VST1    [V26.S4], (R17)
    ADD     $16, R17
    VST1    [V30.S4], (R17)
    ADD     $16, R17
    VST1    [V2.S4], (R17)
    ADD     $16, R17
    // Column p+3 (offset 192): V23, V27, V31, V3
    VST1    [V23.S4], (R17)
    ADD     $16, R17
    VST1    [V27.S4], (R17)
    ADD     $16, R17
    VST1    [V31.S4], (R17)
    ADD     $16, R17
    VST1    [V3.S4], (R17)
    ADD     $16, R17

    SUB     $4, R16, R16
    CMP     $4, R16
    BGE     loop4

    // Scalar tail for remaining columns
tail1:
    CBZ     R16, done

tail_loop:
    FMOVS   (R0), F0
    FMOVS   (R1), F1
    FMOVS   (R2), F2
    FMOVS   (R3), F3
    VMOV    V0.S[0], V16.S[0]
    VMOV    V1.S[0], V16.S[1]
    VMOV    V2.S[0], V16.S[2]
    VMOV    V3.S[0], V16.S[3]
    FMOVS   (R4), F0
    FMOVS   (R5), F1
    FMOVS   (R6), F2
    FMOVS   (R7), F3
    VMOV    V0.S[0], V17.S[0]
    VMOV    V1.S[0], V17.S[1]
    VMOV    V2.S[0], V17.S[2]
    VMOV    V3.S[0], V17.S[3]
    FMOVS   (R8), F0
    FMOVS   (R9), F1
    FMOVS   (R10), F2
    FMOVS   (R11), F3
    VMOV    V0.S[0], V18.S[0]
    VMOV    V1.S[0], V18.S[1]
    VMOV    V2.S[0], V18.S[2]
    VMOV    V3.S[0], V18.S[3]
    FMOVS   (R12), F0
    FMOVS   (R13), F1
    FMOVS   (R14), F2
    FMOVS   (R15), F3
    VMOV    V0.S[0], V19.S[0]
    VMOV    V1.S[0], V19.S[1]
    VMOV    V2.S[0], V19.S[2]
    VMOV    V3.S[0], V19.S[3]
    VST1.P  [V16.S4, V17.S4, V18.S4, V19.S4], 64(R17)
    ADD     $4, R0
    ADD     $4, R1
    ADD     $4, R2
    ADD     $4, R3
    ADD     $4, R4
    ADD     $4, R5
    ADD     $4, R6
    ADD     $4, R7
    ADD     $4, R8
    ADD     $4, R9
    ADD     $4, R10
    ADD     $4, R11
    ADD     $4, R12
    ADD     $4, R13
    ADD     $4, R14
    ADD     $4, R15
    SUB     $1, R16, R16
    CBNZ    R16, tail_loop

done:
    RET
