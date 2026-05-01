package tensor

import "math"

// StandardRules returns the default set of algebraic simplification rules.
func StandardRules() *PatternMatcher {
	return NewPatternMatcher(
		// --- Constant folding: CONST op CONST → CONST ---
		RewriteRule{
			Pat: Pat(OpAdd).WithSrc(ConstPat("a"), ConstPat("b")),
			Rewrite: constFold(func(a, b float64) float64 { return a + b }),
		},
		RewriteRule{
			Pat: Pat(OpSub).WithSrc(ConstPat("a"), ConstPat("b")),
			Rewrite: constFold(func(a, b float64) float64 { return a - b }),
		},
		RewriteRule{
			Pat: Pat(OpMul).WithSrc(ConstPat("a"), ConstPat("b")),
			Rewrite: constFold(func(a, b float64) float64 { return a * b }),
		},
		RewriteRule{
			Pat: Pat(OpDiv).WithSrc(ConstPat("a"), ConstPat("b")),
			Rewrite: constFold(func(a, b float64) float64 { return a / b }),
		},

		// --- Unary constant folding ---
		RewriteRule{
			Pat: Pat(OpNeg).WithSrc(ConstPat("a")),
			Rewrite: unaryConstFold(func(a float64) float64 { return -a }),
		},
		RewriteRule{
			Pat: Pat(OpSqrt).WithSrc(ConstPat("a")),
			Rewrite: unaryConstFold(func(a float64) float64 { return math.Sqrt(a) }),
		},
		RewriteRule{
			Pat: Pat(OpExp2).WithSrc(ConstPat("a")),
			Rewrite: unaryConstFold(func(a float64) float64 { return math.Exp2(a) }),
		},
		RewriteRule{
			Pat: Pat(OpLog2).WithSrc(ConstPat("a")),
			Rewrite: unaryConstFold(func(a float64) float64 { return math.Log2(a) }),
		},

		// --- Algebraic identities ---
		// x + 0 → x
		RewriteRule{
			Pat: Pat(OpAdd).WithSrc(Var("x"), constZero()),
			Rewrite: func(b map[string]*UOp, _ *UOp) *UOp { return b["x"] },
		},
		// 0 + x → x
		RewriteRule{
			Pat: Pat(OpAdd).WithSrc(constZero(), Var("x")),
			Rewrite: func(b map[string]*UOp, _ *UOp) *UOp { return b["x"] },
		},
		// x * 1 → x
		RewriteRule{
			Pat: Pat(OpMul).WithSrc(Var("x"), constOne()),
			Rewrite: func(b map[string]*UOp, _ *UOp) *UOp { return b["x"] },
		},
		// 1 * x → x
		RewriteRule{
			Pat: Pat(OpMul).WithSrc(constOne(), Var("x")),
			Rewrite: func(b map[string]*UOp, _ *UOp) *UOp { return b["x"] },
		},
		// x * 0 → 0
		RewriteRule{
			Pat: Pat(OpMul).WithSrc(Var("_"), constZero()),
			Rewrite: func(b map[string]*UOp, u *UOp) *UOp { return ConstOp(u.DType, 0) },
		},
		// 0 * x → 0
		RewriteRule{
			Pat: Pat(OpMul).WithSrc(constZero(), Var("_")),
			Rewrite: func(b map[string]*UOp, u *UOp) *UOp { return ConstOp(u.DType, 0) },
		},
		// x - 0 → x
		RewriteRule{
			Pat: Pat(OpSub).WithSrc(Var("x"), constZero()),
			Rewrite: func(b map[string]*UOp, _ *UOp) *UOp { return b["x"] },
		},
		// x / 1 → x
		RewriteRule{
			Pat: Pat(OpDiv).WithSrc(Var("x"), constOne()),
			Rewrite: func(b map[string]*UOp, _ *UOp) *UOp { return b["x"] },
		},
		// --x → x (double negation)
		RewriteRule{
			Pat: Pat(OpNeg).WithSrc(Pat(OpNeg).WithSrc(Var("x"))),
			Rewrite: func(b map[string]*UOp, _ *UOp) *UOp { return b["x"] },
		},
		// reshape(reshape(x)) → reshape(x) (movement fusion)
		RewriteRule{
			Pat: Pat(OpReshape).WithSrc(Pat(OpReshape).WithSrc(Var("x"))),
			Rewrite: func(b map[string]*UOp, u *UOp) *UOp {
				return newUOp(OpReshape, u.DType, []*UOp{b["x"]}, u.Arg)
			},
		},
	)
}

// --- Helpers ---

func constZero() *UPat {
	return &UPat{Ops: []Ops{OpConst}, ArgMatch: func(a any) bool {
		v, ok := a.(float64)
		return ok && v == 0
	}}
}

func constOne() *UPat {
	return &UPat{Ops: []Ops{OpConst}, ArgMatch: func(a any) bool {
		v, ok := a.(float64)
		return ok && v == 1
	}}
}

func constFold(f func(float64, float64) float64) func(map[string]*UOp, *UOp) *UOp {
	return func(b map[string]*UOp, u *UOp) *UOp {
		a := b["a"].Arg.(float64)
		bv := b["b"].Arg.(float64)
		return ConstOp(u.DType, f(a, bv))
	}
}

func unaryConstFold(f func(float64) float64) func(map[string]*UOp, *UOp) *UOp {
	return func(b map[string]*UOp, u *UOp) *UOp {
		a := b["a"].Arg.(float64)
		return ConstOp(u.DType, f(a))
	}
}
