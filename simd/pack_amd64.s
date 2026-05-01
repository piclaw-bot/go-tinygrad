// pack_amd64.s — AVX2 pack: transpose 16 B rows into column-panel format
// func packBNTAsm(b0..b15 uintptr, k int, bp uintptr)

#include "textflag.h"

TEXT ·packBNTAsm(SB), NOSPLIT, $0-144
    // Load 16 row pointers. We only have 14 GP regs (AX,BX,CX,DX,SI,DI,R8-R15).
    // Use stack reloads for rows 14,15.
    MOVQ    b0+0(FP), AX
    MOVQ    b1+8(FP), BX
    MOVQ    b2+16(FP), DX
    MOVQ    b3+24(FP), SI
    MOVQ    b4+32(FP), R8
    MOVQ    b5+40(FP), R9
    MOVQ    b6+48(FP), R10
    MOVQ    b7+56(FP), R11
    MOVQ    b8+64(FP), R12
    MOVQ    b9+72(FP), R13
    MOVQ    b10+80(FP), R14
    MOVQ    b11+88(FP), R15
    // R12-R15 = rows 8-11, rows 12-15 loaded per iteration from stack
    MOVQ    k+128(FP), CX
    MOVQ    bp+136(FP), DI

    TESTQ   CX, CX
    JZ      done

loop:
    // Group 0: rows 0-3 → X0
    VMOVSS  (AX), X0
    VINSERTPS $0x10, (BX), X0, X0
    VINSERTPS $0x20, (DX), X0, X0
    VINSERTPS $0x30, (SI), X0, X0
    // Group 1: rows 4-7 → X1
    VMOVSS  (R8), X1
    VINSERTPS $0x10, (R9), X1, X1
    VINSERTPS $0x20, (R10), X1, X1
    VINSERTPS $0x30, (R11), X1, X1
    // Group 2: rows 8-11 → X2
    VMOVSS  (R12), X2
    VINSERTPS $0x10, (R13), X2, X2
    VINSERTPS $0x20, (R14), X2, X2
    VINSERTPS $0x30, (R15), X2, X2
    // Group 3: rows 12-15 → X3 (reload from stack)
    PUSHQ   AX
    MOVQ    b12+96(FP), AX
    VMOVSS  (AX), X3
    ADDQ    $4, AX
    MOVQ    AX, b12+96(FP)
    MOVQ    b13+104(FP), AX
    VINSERTPS $0x10, (AX), X3, X3
    ADDQ    $4, AX
    MOVQ    AX, b13+104(FP)
    MOVQ    b14+112(FP), AX
    VINSERTPS $0x20, (AX), X3, X3
    ADDQ    $4, AX
    MOVQ    AX, b14+112(FP)
    MOVQ    b15+120(FP), AX
    VINSERTPS $0x30, (AX), X3, X3
    ADDQ    $4, AX
    MOVQ    AX, b15+120(FP)
    POPQ    AX

    // Store 16 floats
    VMOVUPS X0, (DI)
    VMOVUPS X1, 16(DI)
    VMOVUPS X2, 32(DI)
    VMOVUPS X3, 48(DI)

    // Advance rows 0-11
    ADDQ    $4, AX
    ADDQ    $4, BX
    ADDQ    $4, DX
    ADDQ    $4, SI
    ADDQ    $4, R8
    ADDQ    $4, R9
    ADDQ    $4, R10
    ADDQ    $4, R11
    ADDQ    $4, R12
    ADDQ    $4, R13
    ADDQ    $4, R14
    ADDQ    $4, R15
    ADDQ    $64, DI

    DECQ    CX
    JNZ     loop

done:
    VZEROUPPER
    RET
