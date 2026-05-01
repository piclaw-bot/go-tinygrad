// simd_arm64.s — ARM64 NEON SIMD kernels for Go (plan9 syntax)
//
// func Sdot(x, y []float32) float32
// func Saxpy(alpha float32, x []float32, y []float32)
//
// Uses VFMLA (FMA, 4x float32) for the main loop.
// VFADD/VFADDP aren't in Go's arm64 assembler yet, so horizontal
// reduction uses scalar FADDS after extracting vector lanes.

#include "textflag.h"

// VFADD V2.S4, V0.S4, V0.S4 — encoded manually (not in Go's assembler)
// Encoding: 0x4e20d440 = FADD V0.4S, V2.4S, V0.4S
#define VFADD_V2_V0_V0 WORD $0x4e22d400
// VFADD V1.S4, V0.S4, V0.S4
#define VFADD_V1_V0_V0 WORD $0x4e21d400

// func Sdot(x, y []float32) float32
TEXT ·Sdot(SB), NOSPLIT, $0-52
    MOVD    x_base+0(FP), R0
    MOVD    x_len+8(FP), R2
    MOVD    y_base+24(FP), R1

    VEOR    V0.B16, V0.B16, V0.B16  // acc0 = 0
    VEOR    V1.B16, V1.B16, V1.B16  // acc1 = 0

    CMP     $8, R2
    BLT     sdot_tail4

sdot_loop8:
    VLD1.P  32(R0), [V4.S4, V5.S4]  // x[i:i+8]
    VLD1.P  32(R1), [V6.S4, V7.S4]  // y[i:i+8]
    VFMLA   V4.S4, V6.S4, V0.S4     // acc0 += x * y
    VFMLA   V5.S4, V7.S4, V1.S4     // acc1 += x * y
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     sdot_loop8

sdot_tail4:
    // acc0 += acc1
    VFADD_V1_V0_V0

    CMP     $4, R2
    BLT     sdot_reduce

    VLD1.P  16(R0), [V4.S4]
    VLD1.P  16(R1), [V6.S4]
    VFMLA   V4.S4, V6.S4, V0.S4
    SUB     $4, R2, R2

sdot_reduce:
    // Horizontal sum of V0.S4: extract 4 lanes to scalar and add
    // V0 = [a, b, c, d]
    VMOV    V0.S[0], R3
    FMOVS   R3, F4
    VMOV    V0.S[1], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4              // F4 = a + b
    VMOV    V0.S[2], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4              // F4 = a + b + c
    VMOV    V0.S[3], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4              // F4 = a + b + c + d

    // Scalar remainder
    CMP     $0, R2
    BEQ     sdot_done

sdot_scalar:
    FMOVS   (R0), F5
    FMOVS   (R1), F6
    FMADDS  F5, F4, F6, F4
    ADD     $4, R0
    ADD     $4, R1
    SUB     $1, R2, R2
    CBNZ    R2, sdot_scalar

sdot_done:
    FMOVS   F4, ret+48(FP)
    RET

// func Saxpy(alpha float32, x []float32, y []float32)
TEXT ·Saxpy(SB), NOSPLIT, $0-56
    FMOVS   alpha+0(FP), F8
    VDUP    V8.S[0], V8.S4          // broadcast alpha
    MOVD    x_base+8(FP), R0
    MOVD    x_len+16(FP), R2
    MOVD    y_base+32(FP), R1

    CMP     $8, R2
    BLT     saxpy_tail4

saxpy_loop8:
    VLD1    (R1), [V0.S4, V1.S4]    // load y
    VLD1.P  32(R0), [V4.S4, V5.S4]  // load x
    VFMLA   V8.S4, V4.S4, V0.S4     // y += alpha * x
    VFMLA   V8.S4, V5.S4, V1.S4
    VST1.P  [V0.S4, V1.S4], 32(R1)  // store y
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     saxpy_loop8

saxpy_tail4:
    CMP     $4, R2
    BLT     saxpy_scalar

    VLD1    (R1), [V0.S4]
    VLD1.P  16(R0), [V4.S4]
    VFMLA   V8.S4, V4.S4, V0.S4
    VST1.P  [V0.S4], 16(R1)
    SUB     $4, R2, R2

saxpy_scalar:
    CMP     $0, R2
    BEQ     saxpy_done

saxpy_scalar_loop:
    FMOVS   (R0), F4
    FMOVS   (R1), F5
    FMADDS  F4, F5, F8, F5
    FMOVS   F5, (R1)
    ADD     $4, R0
    ADD     $4, R1
    SUB     $1, R2, R2
    CBNZ    R2, saxpy_scalar_loop

saxpy_done:
    RET
