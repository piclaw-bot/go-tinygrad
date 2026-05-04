// simd/vec_arm64.s — ARM64 NEON vector operations for inference
//
// All functions process 8 floats/iteration (2× V registers), 4-wide tail, scalar remainder.

#include "textflag.h"

// FADD V-form macros (not in Go's arm64 assembler)
#define VFADD_V1_V0_V0 WORD $0x4e21d400
#define VFADD_V2_V0_V0 WORD $0x4e22d400
#define VFADD_V3_V0_V0 WORD $0x4e23d400

// func Snrm2(x []float32) float32
TEXT ·Snrm2(SB), NOSPLIT, $0-28
    MOVD    x_base+0(FP), R0
    MOVD    x_len+8(FP), R2
    VEOR    V0.B16, V0.B16, V0.B16
    VEOR    V1.B16, V1.B16, V1.B16

    CMP     $8, R2
    BLT     snrm2_tail4

snrm2_loop8:
    VLD1.P  32(R0), [V4.S4, V5.S4]
    VFMLA   V4.S4, V4.S4, V0.S4
    VFMLA   V5.S4, V5.S4, V1.S4
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     snrm2_loop8

snrm2_tail4:
    VFADD_V1_V0_V0
    CMP     $4, R2
    BLT     snrm2_reduce
    VLD1.P  16(R0), [V4.S4]
    VFMLA   V4.S4, V4.S4, V0.S4
    SUB     $4, R2, R2

snrm2_reduce:
    VMOV    V0.S[0], R3
    FMOVS   R3, F4
    VMOV    V0.S[1], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4
    VMOV    V0.S[2], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4
    VMOV    V0.S[3], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4

    CMP     $0, R2
    BEQ     snrm2_sqrt

snrm2_scalar:
    FMOVS   (R0), F5
    FMADDS  F5, F4, F5, F4
    ADD     $4, R0
    SUB     $1, R2, R2
    CBNZ    R2, snrm2_scalar

snrm2_sqrt:
    FSQRTS  F4, F4
    FMOVS   F4, ret+24(FP)
    RET

// func VecAdd(dst, a, b []float32)
TEXT ·VecAdd(SB), NOSPLIT, $0-72
    MOVD    dst_base+0(FP), R3
    MOVD    a_base+24(FP), R0
    MOVD    a_len+32(FP), R2
    MOVD    b_base+48(FP), R1

    CMP     $8, R2
    BLT     vadd_tail4

vadd_loop8:
    VLD1.P  32(R0), [V0.S4, V1.S4]
    VLD1.P  32(R1), [V2.S4, V3.S4]
    WORD    $0x4e22d400  // FADD V0.4S, V0.4S, V2.4S
    WORD    $0x4e23d421  // FADD V1.4S, V1.4S, V3.4S
    VST1.P  [V0.S4, V1.S4], 32(R3)
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     vadd_loop8

vadd_tail4:
    CMP     $4, R2
    BLT     vadd_scalar
    VLD1.P  16(R0), [V0.S4]
    VLD1.P  16(R1), [V2.S4]
    WORD    $0x4e22d400
    VST1.P  [V0.S4], 16(R3)
    SUB     $4, R2, R2

vadd_scalar:
    CMP     $0, R2
    BEQ     vadd_done

vadd_scalar_loop:
    FMOVS   (R0), F0
    FMOVS   (R1), F1
    FADDS   F1, F0, F0
    FMOVS   F0, (R3)
    ADD     $4, R0
    ADD     $4, R1
    ADD     $4, R3
    SUB     $1, R2, R2
    CBNZ    R2, vadd_scalar_loop

vadd_done:
    RET

// func VecMul(dst, a, b []float32)
TEXT ·VecMul(SB), NOSPLIT, $0-72
    MOVD    dst_base+0(FP), R3
    MOVD    a_base+24(FP), R0
    MOVD    a_len+32(FP), R2
    MOVD    b_base+48(FP), R1

    CMP     $8, R2
    BLT     vmul_tail4

vmul_loop8:
    VLD1.P  32(R0), [V0.S4, V1.S4]
    VLD1.P  32(R1), [V2.S4, V3.S4]
    WORD    $0x6e22dc00  // FMUL V0.4S, V0.4S, V2.4S
    WORD    $0x6e23dc21  // FMUL V1.4S, V1.4S, V3.4S
    VST1.P  [V0.S4, V1.S4], 32(R3)
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     vmul_loop8

vmul_tail4:
    CMP     $4, R2
    BLT     vmul_scalar
    VLD1.P  16(R0), [V0.S4]
    VLD1.P  16(R1), [V2.S4]
    WORD    $0x6e22dc00
    VST1.P  [V0.S4], 16(R3)
    SUB     $4, R2, R2

vmul_scalar:
    CMP     $0, R2
    BEQ     vmul_done

vmul_scalar_loop:
    FMOVS   (R0), F0
    FMOVS   (R1), F1
    FMULS   F1, F0, F0
    FMOVS   F0, (R3)
    ADD     $4, R0
    ADD     $4, R1
    ADD     $4, R3
    SUB     $1, R2, R2
    CBNZ    R2, vmul_scalar_loop

vmul_done:
    RET

// func VecScaleAdd(dst, a, b []float32, scale float32)
TEXT ·VecScaleAdd(SB), NOSPLIT, $0-76
    MOVD    dst_base+0(FP), R3
    MOVD    a_base+24(FP), R0
    MOVD    a_len+32(FP), R2
    MOVD    b_base+48(FP), R1
    FMOVS   scale+72(FP), F8
    VDUP    V8.S[0], V8.S4

    CMP     $8, R2
    BLT     vsa_tail4

vsa_loop8:
    VLD1.P  32(R0), [V0.S4, V1.S4]
    VLD1.P  32(R1), [V4.S4, V5.S4]
    VFMLA   V8.S4, V4.S4, V0.S4
    VFMLA   V8.S4, V5.S4, V1.S4
    VST1.P  [V0.S4, V1.S4], 32(R3)
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     vsa_loop8

vsa_tail4:
    CMP     $4, R2
    BLT     vsa_scalar
    VLD1.P  16(R0), [V0.S4]
    VLD1.P  16(R1), [V4.S4]
    VFMLA   V8.S4, V4.S4, V0.S4
    VST1.P  [V0.S4], 16(R3)
    SUB     $4, R2, R2

vsa_scalar:
    CMP     $0, R2
    BEQ     vsa_done

vsa_scalar_loop:
    FMOVS   (R0), F0
    FMOVS   (R1), F4
    FMADDS  F4, F0, F8, F0
    FMOVS   F0, (R3)
    ADD     $4, R0
    ADD     $4, R1
    ADD     $4, R3
    SUB     $1, R2, R2
    CBNZ    R2, vsa_scalar_loop

vsa_done:
    RET

// func RMSNorm(x, w []float32, eps float32)
TEXT ·RMSNorm(SB), NOSPLIT, $0-52
    MOVD    x_base+0(FP), R0
    MOVD    x_len+8(FP), R2
    MOVD    w_base+24(FP), R1
    FMOVS   eps+48(FP), F8

    MOVD    R0, R4          // save x
    MOVD    R2, R5          // save n

    // Phase 1: sum of squares
    VEOR    V0.B16, V0.B16, V0.B16
    VEOR    V1.B16, V1.B16, V1.B16

    CMP     $8, R2
    BLT     rn_ss_tail4

rn_ss_loop8:
    VLD1.P  32(R0), [V4.S4, V5.S4]
    VFMLA   V4.S4, V4.S4, V0.S4
    VFMLA   V5.S4, V5.S4, V1.S4
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     rn_ss_loop8

rn_ss_tail4:
    VFADD_V1_V0_V0
    CMP     $4, R2
    BLT     rn_ss_reduce
    VLD1.P  16(R0), [V4.S4]
    VFMLA   V4.S4, V4.S4, V0.S4
    SUB     $4, R2, R2

rn_ss_reduce:
    VMOV    V0.S[0], R3
    FMOVS   R3, F4
    VMOV    V0.S[1], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4
    VMOV    V0.S[2], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4
    VMOV    V0.S[3], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4

    CMP     $0, R2
    BEQ     rn_compute

rn_ss_scalar:
    FMOVS   (R0), F5
    FMADDS  F5, F4, F5, F4
    ADD     $4, R0
    SUB     $1, R2, R2
    CBNZ    R2, rn_ss_scalar

rn_compute:
    // F4 = sum_sq, R5 = n
    SCVTFS  R5, F5           // F5 = float(n)
    FDIVS   F5, F4, F4       // F4 = sum/n
    FADDS   F8, F4, F4       // F4 = sum/n + eps
    FSQRTS  F4, F4           // F4 = sqrt(...)
    FMOVS   $1.0, F5
    FDIVS   F4, F5, F4       // F4 = invRMS
    VDUP    V6.S[0], V6.S4   // Need to put F4 into V6
    FMOVS   F4, R3
    VMOV    R3, V6.S[0]
    VDUP    V6.S[0], V6.S4   // broadcast invRMS

    // Phase 2: x[i] = w[i] * x[i] * invRMS
    MOVD    R4, R0
    MOVD    R5, R2

    CMP     $8, R2
    BLT     rn_apply_tail4

rn_apply_loop8:
    VLD1    (R0), [V0.S4, V1.S4]
    VLD1.P  32(R1), [V4.S4, V5.S4]
    WORD    $0x6e26dc00  // FMUL V0.4S, V0.4S, V6.4S (x*invRMS)
    WORD    $0x6e26dc21  // FMUL V1.4S, V1.4S, V6.4S
    WORD    $0x6e24dc00  // FMUL V0.4S, V0.4S, V4.4S (*w)
    WORD    $0x6e25dc21  // FMUL V1.4S, V1.4S, V5.4S
    VST1.P  [V0.S4, V1.S4], 32(R0)
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     rn_apply_loop8

rn_apply_tail4:
    CMP     $4, R2
    BLT     rn_apply_scalar
    VLD1    (R0), [V0.S4]
    VLD1.P  16(R1), [V4.S4]
    WORD    $0x6e26dc00
    WORD    $0x6e24dc00
    VST1.P  [V0.S4], 16(R0)
    SUB     $4, R2, R2

rn_apply_scalar:
    CMP     $0, R2
    BEQ     rn_done

rn_apply_scalar_loop:
    FMOVS   (R0), F0
    FMOVS   (R1), F5
    FMULS   F4, F0, F0       // x * invRMS
    FMULS   F5, F0, F0       // * w
    FMOVS   F0, (R0)
    ADD     $4, R0
    ADD     $4, R1
    SUB     $1, R2, R2
    CBNZ    R2, rn_apply_scalar_loop

rn_done:
    RET

// func RMSNormBF16(x, w []float32, eps float32)
// Same as RMSNorm + BF16 truncate on output
TEXT ·RMSNormBF16(SB), NOSPLIT, $0-52
    // For ARM64, just call RMSNorm then ToBF16
    // (ARM64 doesn't have a convenient BF16 AND mask like AVX2 VANDPS)
    MOVD    x_base+0(FP), R0
    MOVD    x_len+8(FP), R2

    // Call RMSNorm first
    BL      ·RMSNorm(SB)

    // Then truncate to BF16
    MOVD    x_base+0(FP), R0
    MOVD    x_len+8(FP), R2
    B       tobf16_entry

// func ToBF16(x []float32)
TEXT ·ToBF16(SB), NOSPLIT, $0-24
    MOVD    x_base+0(FP), R0
    MOVD    x_len+8(FP), R2

tobf16_entry:
    // BF16: mask = 0xFFFF0000 per 32-bit word
    MOVW    $0xFFFF0000, R3
    VMOV    R3, V7.S[0]
    VDUP    V7.S[0], V7.S4

    CMP     $8, R2
    BLT     bf16_tail4

bf16_loop8:
    VLD1    (R0), [V0.S4, V1.S4]
    WORD    $0x4e271c00  // AND V0.16B, V0.16B, V7.16B
    WORD    $0x4e271c21  // AND V1.16B, V1.16B, V7.16B
    VST1.P  [V0.S4, V1.S4], 32(R0)
    SUB     $8, R2, R2
    CMP     $8, R2
    BGE     bf16_loop8

bf16_tail4:
    CMP     $4, R2
    BLT     bf16_scalar
    VLD1    (R0), [V0.S4]
    WORD    $0x4e271c00
    VST1.P  [V0.S4], 16(R0)
    SUB     $4, R2, R2

bf16_scalar:
    CMP     $0, R2
    BEQ     bf16_done

bf16_scalar_loop:
    MOVW    (R0), R4
    AND     $0xFFFF0000, R4, R4
    MOVW    R4, (R0)
    ADD     $4, R0
    SUB     $1, R2, R2
    CBNZ    R2, bf16_scalar_loop

bf16_done:
    RET

// func VecSiLUMul(dst, a, b []float32)
TEXT ·VecSiLUMul(SB), NOSPLIT, $0-72
    B       ·vecSiLUMulGo(SB)

// ============================================================
// BF16 SIMD operations (ARM64 NEON)
// Widen: USHLL (zero-extend u16→u32) + SHL $16
// Narrow: USHR $16 + UZP1/XTN
// ============================================================

// func BF16DotAsm(x, y []uint16) float32
TEXT ·BF16DotAsm(SB), NOSPLIT, $0-52
    MOVD    x_base+0(FP), R0
    MOVD    x_len+8(FP), R2
    MOVD    y_base+24(FP), R1

    VEOR    V0.B16, V0.B16, V0.B16   // acc

    CMP     $4, R2
    BLT     bf16dot_arm_scalar

bf16dot_arm_loop4:
    // Load 4× BF16
    VLD1    (R0), [V2.H4]           // 4× u16
    VLD1    (R1), [V3.H4]
    // Widen to u32 then shift to F32
    WORD    $0x2f10a442              // USHLL V2.4S, V2.4H, #0 (zero-extend)
    WORD    $0x4f305442              // SHL V2.4S, V2.4S, #16
    WORD    $0x2f10a463
    WORD    $0x4f305463
    // FMA
    VFMLA   V2.S4, V3.S4, V0.S4
    ADD     $8, R0
    ADD     $8, R1
    SUB     $4, R2, R2
    CMP     $4, R2
    BGE     bf16dot_arm_loop4

bf16dot_arm_scalar:
    // Horizontal reduce V0
    VMOV    V0.S[0], R3
    FMOVS   R3, F4
    VMOV    V0.S[1], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4
    VMOV    V0.S[2], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4
    VMOV    V0.S[3], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4

    CMP     $0, R2
    BEQ     bf16dot_arm_done

bf16dot_arm_tail:
    MOVHU   (R0), R3
    LSL     $16, R3, R3
    FMOVS   R3, F5
    MOVHU   (R1), R3
    LSL     $16, R3, R3
    FMOVS   R3, F6
    FMADDS  F5, F4, F6, F4
    ADD     $2, R0
    ADD     $2, R1
    SUB     $1, R2, R2
    CBNZ    R2, bf16dot_arm_tail

bf16dot_arm_done:
    FMOVS   F4, ret+48(FP)
    RET

// func BF16VecAddAsm(dst, a, b []uint16)
// Widen BF16→F32, add, narrow F32→BF16
TEXT ·BF16VecAddAsm(SB), NOSPLIT, $0-72
    MOVD    dst_base+0(FP), R3
    MOVD    a_base+24(FP), R0
    MOVD    a_len+32(FP), R2
    MOVD    b_base+48(FP), R1

    CMP     $4, R2
    BLT     bf16add_arm_scalar

bf16add_arm_loop4:
    // Load 4× BF16 from a and b
    VLD1    (R0), [V2.H4]
    VLD1    (R1), [V3.H4]
    // Widen to F32: USHLL (u16→u32) + SHL #16
    WORD    $0x2f10a442   // USHLL V2.4S, V2.4H, #0
    WORD    $0x4f305442   // SHL   V2.4S, V2.4S, #16
    WORD    $0x2f10a463   // USHLL V3.4S, V3.4H, #0
    WORD    $0x4f305463   // SHL   V3.4S, V3.4S, #16
    // F32 add
    WORD    $0x4e23d442   // FADD  V2.4S, V2.4S, V3.4S
    // Narrow F32→BF16: USHR #16 + XTN
    WORD    $0x6f300442   // USHR  V2.4S, V2.4S, #16
    WORD    $0x0ea12842   // XTN   V2.4H, V2.4S
    VST1    [V2.H4], (R3)
    ADD     $8, R0
    ADD     $8, R1
    ADD     $8, R3
    SUB     $4, R2, R2
    CMP     $4, R2
    BGE     bf16add_arm_loop4

bf16add_arm_scalar:
    CMP     $0, R2
    BEQ     bf16add_arm_done

bf16add_arm_scalar_loop:
    MOVHU   (R0), R4
    LSL     $16, R4, R4
    FMOVS   R4, F0
    MOVHU   (R1), R4
    LSL     $16, R4, R4
    FMOVS   R4, F1
    FADDS   F1, F0, F0
    FMOVS   F0, R4
    LSR     $16, R4, R4
    MOVH    R4, (R3)
    ADD     $2, R0
    ADD     $2, R1
    ADD     $2, R3
    SUB     $1, R2, R2
    CBNZ    R2, bf16add_arm_scalar_loop

bf16add_arm_done:
    RET

// func BF16RMSNormAsm(x, w []uint16, eps float32)
// Phase 1: widen→square→sum. Phase 2: widen→scale→narrow.
TEXT ·BF16RMSNormAsm(SB), NOSPLIT, $0-52
    MOVD    x_base+0(FP), R0
    MOVD    x_len+8(FP), R2
    MOVD    w_base+24(FP), R1
    FMOVS   eps+48(FP), F8
    MOVD    R0, R4          // save x
    MOVD    R2, R5          // save n

    // Phase 1: sum of squares
    VEOR    V0.B16, V0.B16, V0.B16

    CMP     $4, R2
    BLT     bf16rn_arm_ss_scalar

bf16rn_arm_ss_loop4:
    VLD1    (R0), [V2.H4]
    WORD    $0x2f10a442    // USHLL V2.4S, V2.4H, #0
    WORD    $0x4f305442    // SHL   V2.4S, V2.4S, #16
    VFMLA   V2.S4, V2.S4, V0.S4
    ADD     $8, R0
    SUB     $4, R2, R2
    CMP     $4, R2
    BGE     bf16rn_arm_ss_loop4

bf16rn_arm_ss_scalar:
    // Horizontal reduce V0
    VMOV    V0.S[0], R3
    FMOVS   R3, F4
    VMOV    V0.S[1], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4
    VMOV    V0.S[2], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4
    VMOV    V0.S[3], R3
    FMOVS   R3, F5
    FADDS   F5, F4, F4

    CMP     $0, R2
    BEQ     bf16rn_arm_compute

bf16rn_arm_ss_tail:
    MOVHU   (R0), R3
    LSL     $16, R3, R3
    FMOVS   R3, F5
    FMADDS  F5, F4, F5, F4
    ADD     $2, R0
    SUB     $1, R2, R2
    CBNZ    R2, bf16rn_arm_ss_tail

bf16rn_arm_compute:
    // F4 = sum_sq
    SCVTFS  R5, F5
    FDIVS   F5, F4, F4
    FADDS   F8, F4, F4
    FSQRTS  F4, F4
    FMOVS   $1.0, F5
    FDIVS   F4, F5, F4     // F4 = invRMS

    // Broadcast invRMS to V6
    FMOVS   F4, R3
    VMOV    R3, V6.S[0]
    VDUP    V6.S[0], V6.S4

    // Phase 2: x[i] = BF16(F32(x[i]) * invRMS * F32(w[i]))
    MOVD    R4, R0
    MOVD    R5, R2

    CMP     $4, R2
    BLT     bf16rn_arm_apply_scalar

bf16rn_arm_apply_loop4:
    VLD1    (R0), [V2.H4]
    VLD1    (R1), [V3.H4]
    WORD    $0x2f10a442
    WORD    $0x4f305442
    WORD    $0x2f10a463
    WORD    $0x4f305463
    WORD    $0x6e26dc42    // FMUL V2.4S, V2.4S, V6.4S
    WORD    $0x6e23dc42    // FMUL V2.4S, V2.4S, V3.4S
    WORD    $0x6f300442    // USHR V2.4S, V2.4S, #16
    WORD    $0x0ea12842    // XTN  V2.4H, V2.4S
    VST1    [V2.H4], (R0)
    ADD     $8, R0
    ADD     $8, R1
    SUB     $4, R2, R2
    CMP     $4, R2
    BGE     bf16rn_arm_apply_loop4

bf16rn_arm_apply_scalar:
    CMP     $0, R2
    BEQ     bf16rn_arm_done

bf16rn_arm_apply_tail:
    MOVHU   (R0), R3
    LSL     $16, R3, R3
    FMOVS   R3, F0
    MOVHU   (R1), R3
    LSL     $16, R3, R3
    FMOVS   R3, F5
    FMULS   F4, F0, F0
    FMULS   F5, F0, F0
    FMOVS   F0, R3
    LSR     $16, R3, R3
    MOVH    R3, (R0)
    ADD     $2, R0
    ADD     $2, R1
    SUB     $1, R2, R2
    CBNZ    R2, bf16rn_arm_apply_tail

bf16rn_arm_done:
    RET

// func BF16WidenToF32(dst []float32, src []uint16)
TEXT ·BF16WidenToF32(SB), NOSPLIT, $0-48
    MOVD    dst_base+0(FP), R3
    MOVD    src_base+24(FP), R0
    MOVD    src_len+32(FP), R2

    CMP     $0, R2
    BEQ     bfw_arm_done

bfw_arm_loop:
    MOVHU   (R0), R4
    LSL     $16, R4, R4
    MOVW    R4, (R3)
    ADD     $2, R0
    ADD     $4, R3
    SUB     $1, R2, R2
    CBNZ    R2, bfw_arm_loop

bfw_arm_done:
    RET

// func BF16NarrowFromF32(dst []uint16, src []float32)
TEXT ·BF16NarrowFromF32(SB), NOSPLIT, $0-48
    MOVD    dst_base+0(FP), R3
    MOVD    src_base+24(FP), R0
    MOVD    src_len+32(FP), R2

    CMP     $0, R2
    BEQ     bfn_arm_done

bfn_arm_loop:
    MOVW    (R0), R4
    LSR     $16, R4, R4
    MOVH    R4, (R3)
    ADD     $4, R0
    ADD     $2, R3
    SUB     $1, R2, R2
    CBNZ    R2, bfn_arm_loop

bfn_arm_done:
    RET
