// simd/vec_amd64.s — AVX2/FMA vector operations for inference
//
// All functions process 16 floats/iteration (2× YMM), 8-wide tail, scalar remainder.
// FMA used for dot products and multiply-add patterns.

#include "textflag.h"

// func Snrm2(x []float32) float32
// Returns sqrt(sum(x[i]^2))
TEXT ·Snrm2(SB), NOSPLIT, $0-28
    MOVQ    x_base+0(FP), SI
    MOVQ    x_len+8(FP), CX
    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    CMPQ    CX, $16
    JL      snrm2_tail8

snrm2_loop16:
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS Y2, Y2, Y0
    VFMADD231PS Y3, Y3, Y1
    ADDQ    $64, SI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     snrm2_loop16

snrm2_tail8:
    VADDPS  Y1, Y0, Y0
    CMPQ    CX, $8
    JL      snrm2_reduce
    VMOVUPS (SI), Y2
    VFMADD231PS Y2, Y2, Y0
    ADDQ    $32, SI
    SUBQ    $8, CX

snrm2_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    CMPQ    CX, $0
    JE      snrm2_sqrt

snrm2_scalar:
    VMOVSS  (SI), X1
    VFMADD231SS X1, X1, X0
    ADDQ    $4, SI
    DECQ    CX
    JNZ     snrm2_scalar

snrm2_sqrt:
    VSQRTSS X0, X0, X0
    VMOVSS  X0, ret+24(FP)
    VZEROUPPER
    RET

// func VecAdd(dst, a, b []float32)
TEXT ·VecAdd(SB), NOSPLIT, $0-72
    MOVQ    dst_base+0(FP), DI
    MOVQ    a_base+24(FP), SI
    MOVQ    a_len+32(FP), CX
    MOVQ    b_base+48(FP), DX

    CMPQ    CX, $16
    JL      vadd_tail8

vadd_loop16:
    VMOVUPS (SI), Y0
    VMOVUPS 32(SI), Y1
    VADDPS  (DX), Y0, Y0
    VADDPS  32(DX), Y1, Y1
    VMOVUPS Y0, (DI)
    VMOVUPS Y1, 32(DI)
    ADDQ    $64, SI
    ADDQ    $64, DX
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     vadd_loop16

vadd_tail8:
    CMPQ    CX, $8
    JL      vadd_scalar_check
    VMOVUPS (SI), Y0
    VADDPS  (DX), Y0, Y0
    VMOVUPS Y0, (DI)
    ADDQ    $32, SI
    ADDQ    $32, DX
    ADDQ    $32, DI
    SUBQ    $8, CX

vadd_scalar_check:
    TESTQ   CX, CX
    JZ      vadd_done

vadd_scalar:
    VMOVSS  (SI), X0
    VADDSS  (DX), X0, X0
    VMOVSS  X0, (DI)
    ADDQ    $4, SI
    ADDQ    $4, DX
    ADDQ    $4, DI
    DECQ    CX
    JNZ     vadd_scalar

vadd_done:
    VZEROUPPER
    RET

// func VecMul(dst, a, b []float32)
TEXT ·VecMul(SB), NOSPLIT, $0-72
    MOVQ    dst_base+0(FP), DI
    MOVQ    a_base+24(FP), SI
    MOVQ    a_len+32(FP), CX
    MOVQ    b_base+48(FP), DX

    CMPQ    CX, $16
    JL      vmul_tail8

vmul_loop16:
    VMOVUPS (SI), Y0
    VMOVUPS 32(SI), Y1
    VMULPS  (DX), Y0, Y0
    VMULPS  32(DX), Y1, Y1
    VMOVUPS Y0, (DI)
    VMOVUPS Y1, 32(DI)
    ADDQ    $64, SI
    ADDQ    $64, DX
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     vmul_loop16

vmul_tail8:
    CMPQ    CX, $8
    JL      vmul_scalar_check
    VMOVUPS (SI), Y0
    VMULPS  (DX), Y0, Y0
    VMOVUPS Y0, (DI)
    ADDQ    $32, SI
    ADDQ    $32, DX
    ADDQ    $32, DI
    SUBQ    $8, CX

vmul_scalar_check:
    TESTQ   CX, CX
    JZ      vmul_done

vmul_scalar:
    VMOVSS  (SI), X0
    VMULSS  (DX), X0, X0
    VMOVSS  X0, (DI)
    ADDQ    $4, SI
    ADDQ    $4, DX
    ADDQ    $4, DI
    DECQ    CX
    JNZ     vmul_scalar

vmul_done:
    VZEROUPPER
    RET

// func VecScaleAdd(dst, a, b []float32, scale float32)
// dst[i] = a[i] + scale * b[i]
TEXT ·VecScaleAdd(SB), NOSPLIT, $0-76
    MOVQ    dst_base+0(FP), DI
    MOVQ    a_base+24(FP), SI
    MOVQ    a_len+32(FP), CX
    MOVQ    b_base+48(FP), DX
    MOVSS   scale+72(FP), X8
    VBROADCASTSS X8, Y8

    CMPQ    CX, $16
    JL      vsa_tail8

vsa_loop16:
    VMOVUPS (SI), Y0
    VMOVUPS 32(SI), Y1
    VMOVUPS (DX), Y2
    VMOVUPS 32(DX), Y3
    VFMADD231PS Y8, Y2, Y0
    VFMADD231PS Y8, Y3, Y1
    VMOVUPS Y0, (DI)
    VMOVUPS Y1, 32(DI)
    ADDQ    $64, SI
    ADDQ    $64, DX
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     vsa_loop16

vsa_tail8:
    CMPQ    CX, $8
    JL      vsa_scalar_check
    VMOVUPS (SI), Y0
    VMOVUPS (DX), Y2
    VFMADD231PS Y8, Y2, Y0
    VMOVUPS Y0, (DI)
    ADDQ    $32, SI
    ADDQ    $32, DX
    ADDQ    $32, DI
    SUBQ    $8, CX

vsa_scalar_check:
    TESTQ   CX, CX
    JZ      vsa_done

vsa_scalar:
    VMOVSS  (SI), X0
    VMOVSS  (DX), X2
    VFMADD231SS X8, X2, X0
    VMOVSS  X0, (DI)
    ADDQ    $4, SI
    ADDQ    $4, DX
    ADDQ    $4, DI
    DECQ    CX
    JNZ     vsa_scalar

vsa_done:
    VZEROUPPER
    RET

// func RMSNorm(x, w []float32, eps float32)
// x[i] = w[i] * x[i] * rsqrt(mean(x^2) + eps)
TEXT ·RMSNorm(SB), NOSPLIT, $0-52
    MOVQ    x_base+0(FP), SI
    MOVQ    x_len+8(FP), CX
    MOVQ    w_base+24(FP), DI
    MOVSS   eps+48(FP), X8

    // Phase 1: compute sum of squares
    MOVQ    SI, R8          // save x pointer
    MOVQ    CX, R9          // save n
    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    CMPQ    CX, $16
    JL      rn_ss_tail8

rn_ss_loop16:
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS Y2, Y2, Y0
    VFMADD231PS Y3, Y3, Y1
    ADDQ    $64, SI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     rn_ss_loop16

rn_ss_tail8:
    VADDPS  Y1, Y0, Y0
    CMPQ    CX, $8
    JL      rn_ss_reduce
    VMOVUPS (SI), Y2
    VFMADD231PS Y2, Y2, Y0
    ADDQ    $32, SI
    SUBQ    $8, CX

rn_ss_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    // Scalar remainder for sum-of-squares
    TESTQ   CX, CX
    JZ      rn_compute

rn_ss_scalar:
    VMOVSS  (SI), X1
    VFMADD231SS X1, X1, X0
    ADDQ    $4, SI
    DECQ    CX
    JNZ     rn_ss_scalar

rn_compute:
    // X0 = sum_of_squares
    // invRMS = 1 / sqrt(sum/n + eps)
    VCVTSI2SSQ R9, X1, X1   // X1 = float(n)
    VDIVSS  X1, X0, X0      // X0 = sum/n
    VADDSS  X8, X0, X0      // X0 = sum/n + eps
    VSQRTSS X0, X0, X0      // X0 = sqrt(sum/n + eps)
    MOVL    $0x3f800000, R11
    MOVL    R11, X1
    VDIVSS  X0, X1, X0      // X0 = 1/sqrt(sum/n + eps) = invRMS
    VBROADCASTSS X0, Y6     // Y6 = invRMS broadcast

    // Phase 2: x[i] = w[i] * x[i] * invRMS
    MOVQ    R8, SI           // restore x pointer
    MOVQ    R9, CX           // restore n

    CMPQ    CX, $16
    JL      rn_apply_tail8

rn_apply_loop16:
    VMOVUPS (SI), Y0
    VMOVUPS 32(SI), Y1
    VMULPS  Y6, Y0, Y0      // x * invRMS
    VMULPS  Y6, Y1, Y1
    VMULPS  (DI), Y0, Y0    // * w
    VMULPS  32(DI), Y1, Y1
    VMOVUPS Y0, (SI)
    VMOVUPS Y1, 32(SI)
    ADDQ    $64, SI
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     rn_apply_loop16

rn_apply_tail8:
    CMPQ    CX, $8
    JL      rn_apply_scalar_check
    VMOVUPS (SI), Y0
    VMULPS  Y6, Y0, Y0
    VMULPS  (DI), Y0, Y0
    VMOVUPS Y0, (SI)
    ADDQ    $32, SI
    ADDQ    $32, DI
    SUBQ    $8, CX

rn_apply_scalar_check:
    TESTQ   CX, CX
    JZ      rn_done

rn_apply_scalar:
    VMOVSS  (SI), X0
    VMULSS  X6, X0, X0
    VMULSS  (DI), X0, X0
    VMOVSS  X0, (SI)
    ADDQ    $4, SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     rn_apply_scalar

rn_done:
    VZEROUPPER
    RET

// func RMSNormBF16(x, w []float32, eps float32)
// Same as RMSNorm but rounds each output to BF16 (mask lower 16 bits)
TEXT ·RMSNormBF16(SB), NOSPLIT, $0-52
    MOVQ    x_base+0(FP), SI
    MOVQ    x_len+8(FP), CX
    MOVQ    w_base+24(FP), DI
    MOVSS   eps+48(FP), X8

    // Phase 1: sum of squares (same as RMSNorm)
    MOVQ    SI, R8
    MOVQ    CX, R9
    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    CMPQ    CX, $16
    JL      rnb_ss_tail

rnb_ss_loop:
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS Y2, Y2, Y0
    VFMADD231PS Y3, Y3, Y1
    ADDQ    $64, SI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     rnb_ss_loop

rnb_ss_tail:
    VADDPS  Y1, Y0, Y0
    CMPQ    CX, $8
    JL      rnb_ss_reduce
    VMOVUPS (SI), Y2
    VFMADD231PS Y2, Y2, Y0
    ADDQ    $32, SI
    SUBQ    $8, CX

rnb_ss_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    TESTQ   CX, CX
    JZ      rnb_compute

rnb_ss_scalar:
    VMOVSS  (SI), X1
    VFMADD231SS X1, X1, X0
    ADDQ    $4, SI
    DECQ    CX
    JNZ     rnb_ss_scalar

rnb_compute:
    VCVTSI2SSQ R9, X1, X1
    VDIVSS  X1, X0, X0
    VADDSS  X8, X0, X0
    VSQRTSS X0, X0, X0
    MOVL    $0x3f800000, R11
    MOVL    R11, X1
    VDIVSS  X0, X1, X0
    VBROADCASTSS X0, Y6

    // BF16 mask: AND with 0xFFFF0000 per element
    MOVL    $0xFFFF0000, R10
    MOVL    R10, X7
    VPBROADCASTD X7, Y7

    // Phase 2: x[i] = toBF16(w[i] * x[i] * invRMS)
    MOVQ    R8, SI
    MOVQ    R9, CX

    CMPQ    CX, $16
    JL      rnb_apply_tail

rnb_apply_loop:
    VMOVUPS (SI), Y0
    VMOVUPS 32(SI), Y1
    VMULPS  Y6, Y0, Y0
    VMULPS  Y6, Y1, Y1
    VMULPS  (DI), Y0, Y0
    VMULPS  32(DI), Y1, Y1
    VANDPS  Y7, Y0, Y0      // BF16 truncate
    VANDPS  Y7, Y1, Y1
    VMOVUPS Y0, (SI)
    VMOVUPS Y1, 32(SI)
    ADDQ    $64, SI
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     rnb_apply_loop

rnb_apply_tail:
    CMPQ    CX, $8
    JL      rnb_apply_scalar_check
    VMOVUPS (SI), Y0
    VMULPS  Y6, Y0, Y0
    VMULPS  (DI), Y0, Y0
    VANDPS  Y7, Y0, Y0
    VMOVUPS Y0, (SI)
    ADDQ    $32, SI
    ADDQ    $32, DI
    SUBQ    $8, CX

rnb_apply_scalar_check:
    TESTQ   CX, CX
    JZ      rnb_done

rnb_apply_scalar:
    VMOVSS  (SI), X0
    VMULSS  X6, X0, X0
    VMULSS  (DI), X0, X0
    MOVL    $0xFFFF0000, R10
    MOVD    R10, X3
    VANDPS  X3, X0, X0
    VMOVSS  X0, (SI)
    ADDQ    $4, SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     rnb_apply_scalar

rnb_done:
    VZEROUPPER
    RET

// func ToBF16(x []float32)
TEXT ·ToBF16(SB), NOSPLIT, $0-24
    MOVQ    x_base+0(FP), SI
    MOVQ    x_len+8(FP), CX

    MOVL    $0xFFFF0000, R10
    MOVL    R10, X7
    VPBROADCASTD X7, Y7

    CMPQ    CX, $16
    JL      bf16_tail8

bf16_loop16:
    VMOVUPS (SI), Y0
    VMOVUPS 32(SI), Y1
    VANDPS  Y7, Y0, Y0
    VANDPS  Y7, Y1, Y1
    VMOVUPS Y0, (SI)
    VMOVUPS Y1, 32(SI)
    ADDQ    $64, SI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     bf16_loop16

bf16_tail8:
    CMPQ    CX, $8
    JL      bf16_scalar_check
    VMOVUPS (SI), Y0
    VANDPS  Y7, Y0, Y0
    VMOVUPS Y0, (SI)
    ADDQ    $32, SI
    SUBQ    $8, CX

bf16_scalar_check:
    TESTQ   CX, CX
    JZ      bf16_done

bf16_scalar:
    MOVL    (SI), R11
    ANDL    $0xFFFF0000, R11
    MOVL    R11, (SI)
    ADDQ    $4, SI
    DECQ    CX
    JNZ     bf16_scalar

bf16_done:
    VZEROUPPER
    RET

// VecSiLUMul is implemented in Go (uses exp which has no simple SIMD form)
// func VecSiLUMul(dst, a, b []float32)
TEXT ·VecSiLUMul(SB), NOSPLIT, $0-72
    JMP ·vecSiLUMulGo(SB)

