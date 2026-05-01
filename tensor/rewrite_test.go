package tensor

import "testing"

func TestPatternMatchConst(t *testing.T) {
	c := ConstOp(Float32, 42.0)
	p := ConstPat("x")
	bindings, ok := p.Match(c)
	if !ok {
		t.Fatal("expected match")
	}
	if bindings["x"] != c {
		t.Fatal("wrong binding")
	}
}

func TestPatternMatchBinary(t *testing.T) {
	a := ConstOp(Float32, 1.0)
	b := ConstOp(Float32, 2.0)
	add := newUOp(OpAdd, Float32, []*UOp{a, b}, nil)

	p := Pat(OpAdd).WithSrc(ConstPat("a"), ConstPat("b"))
	bindings, ok := p.Match(add)
	if !ok {
		t.Fatal("expected match")
	}
	if bindings["a"].Arg.(float64) != 1.0 {
		t.Fatalf("a=%v", bindings["a"].Arg)
	}
	if bindings["b"].Arg.(float64) != 2.0 {
		t.Fatalf("b=%v", bindings["b"].Arg)
	}
}

func TestConstantFolding(t *testing.T) {
	a := ConstOp(Float32, 3.0)
	b := ConstOp(Float32, 4.0)
	add := newUOp(OpAdd, Float32, []*UOp{a, b}, nil)

	pm := StandardRules()
	result := GraphRewrite(add, pm)
	if result.Op != OpConst {
		t.Fatalf("expected CONST, got %s", result.Op)
	}
	if result.Arg.(float64) != 7.0 {
		t.Fatalf("expected 7, got %v", result.Arg)
	}
}

func TestAlgebraicIdentities(t *testing.T) {
	pm := StandardRules()

	// x + 0 → x
	x := BufferOp(Float32, []int{3})
	zero := ConstOp(Float32, 0)
	add := newUOp(OpAdd, Float32, []*UOp{x, zero}, nil)
	result := GraphRewrite(add, pm)
	if result != x {
		t.Fatalf("x+0: expected x, got %s", result.Op)
	}

	// x * 1 → x
	one := ConstOp(Float32, 1)
	mul := newUOp(OpMul, Float32, []*UOp{x, one}, nil)
	result = GraphRewrite(mul, pm)
	if result != x {
		t.Fatalf("x*1: expected x, got %s", result.Op)
	}

	// x * 0 → 0
	mul0 := newUOp(OpMul, Float32, []*UOp{x, zero}, nil)
	result = GraphRewrite(mul0, pm)
	if result.Op != OpConst || result.Arg.(float64) != 0 {
		t.Fatalf("x*0: expected CONST(0), got %s(%v)", result.Op, result.Arg)
	}
}

func TestDoubleNegation(t *testing.T) {
	pm := StandardRules()
	x := BufferOp(Float32, []int{3})
	neg1 := newUOp(OpNeg, Float32, []*UOp{x}, nil)
	neg2 := newUOp(OpNeg, Float32, []*UOp{neg1}, nil)
	result := GraphRewrite(neg2, pm)
	if result != x {
		t.Fatalf("--x: expected x, got %s", result.Op)
	}
}

func TestChainedConstFolding(t *testing.T) {
	pm := StandardRules()
	// (2 + 3) * 4 → 20
	a := ConstOp(Float32, 2)
	b := ConstOp(Float32, 3)
	c := ConstOp(Float32, 4)
	add := newUOp(OpAdd, Float32, []*UOp{a, b}, nil)
	mul := newUOp(OpMul, Float32, []*UOp{add, c}, nil)
	result := GraphRewrite(mul, pm)
	if result.Op != OpConst || result.Arg.(float64) != 20 {
		t.Fatalf("(2+3)*4: expected 20, got %s(%v)", result.Op, result.Arg)
	}
}

func TestReshapeReshape(t *testing.T) {
	pm := StandardRules()
	x := BufferOp(Float32, []int{6})
	r1 := newUOp(OpReshape, Float32, []*UOp{x}, []int{2, 3})
	r2 := newUOp(OpReshape, Float32, []*UOp{r1}, []int{3, 2})
	result := GraphRewrite(r2, pm)
	if result.Op != OpReshape || len(result.Src) != 1 || result.Src[0] != x {
		t.Fatalf("reshape(reshape(x)): expected reshape(x), got %s", result.Op)
	}
}
