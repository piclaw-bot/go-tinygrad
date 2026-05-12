package tensor

import (
	"math"
	"testing"
)

func approx(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func TestFromFloat32(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	x := FromFloat32(data, []int{2, 3})
	if x.Numel() != 6 {
		t.Fatalf("numel=%d want 6", x.Numel())
	}
	got := x.Data()
	for i, v := range got {
		if v != data[i] {
			t.Fatalf("data[%d]=%v want %v", i, v, data[i])
		}
	}
}

func TestAdd(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3}, []int{3})
	b := FromFloat32([]float32{4, 5, 6}, []int{3})
	c := a.Add(b)
	got := c.Data()
	want := []float32{5, 7, 9}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("c[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestMul(t *testing.T) {
	a := FromFloat32([]float32{2, 3, 4}, []int{3})
	b := FromFloat32([]float32{5, 6, 7}, []int{3})
	c := a.Mul(b)
	got := c.Data()
	want := []float32{10, 18, 28}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("c[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestChainedOps(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3}, []int{3})
	b := FromFloat32([]float32{4, 5, 6}, []int{3})
	// (a + b) * a = [5, 14, 27]
	c := a.Add(b).Mul(a)
	got := c.Data()
	want := []float32{5, 14, 27}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("c[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestNeg(t *testing.T) {
	a := FromFloat32([]float32{1, -2, 3}, []int{3})
	got := a.Neg().Data()
	want := []float32{-1, 2, -3}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestSqrt(t *testing.T) {
	a := FromFloat32([]float32{1, 4, 9, 16}, []int{4})
	got := a.Sqrt().Data()
	want := []float32{1, 2, 3, 4}
	for i := range got {
		if !approx(got[i], want[i], 1e-6) {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestReduceSum(t *testing.T) {
	// 2×3 matrix, sum over axis 1
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	got := a.Sum(1).Data()
	// [1+2+3, 4+5+6] = [6, 15]
	want := []float32{6, 15}
	for i := range got {
		if !approx(got[i], want[i], 1e-5) {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestReduceMax(t *testing.T) {
	a := FromFloat32([]float32{1, 5, 3, 4, 2, 6}, []int{2, 3})
	got := a.ReduceMax(1).Data()
	want := []float32{5, 6}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestReshape(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := a.Reshape([]int{3, 2})
	if b.Shape()[0] != 3 || b.Shape()[1] != 2 {
		t.Fatalf("shape=%v want [3 2]", b.Shape())
	}
	got := b.Data()
	want := []float32{1, 2, 3, 4, 5, 6}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestZerosOnes(t *testing.T) {
	z := Zeros([]int{2, 3})
	for _, v := range z.Data() {
		if v != 0 {
			t.Fatal("zeros contains non-zero")
		}
	}
	o := Ones([]int{2, 3})
	for _, v := range o.Data() {
		if v != 1 {
			t.Fatal("ones contains non-one")
		}
	}
}

func TestLazyEvaluation(t *testing.T) {
	a := FromFloat32([]float32{1, 2}, []int{2})
	b := FromFloat32([]float32{3, 4}, []int{2})
	c := a.Add(b) // lazy — no computation yet
	if c.uop.buf != nil {
		t.Fatal("expected lazy (nil buf)")
	}
	c.Realize() // triggers computation
	if c.uop.buf == nil {
		t.Fatal("expected realized (non-nil buf)")
	}
	got := c.Data()
	if got[0] != 4 || got[1] != 6 {
		t.Fatalf("got %v want [4 6]", got)
	}
}

func BenchmarkAddMul(b *testing.B) {
	n := 1024 * 1024
	x := Rand([]int{n})
	y := Rand([]int{n})
	x.Realize()
	y.Realize()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z := x.Add(y).Mul(x)
		z.Realize()
	}
}

func TestMatMul(t *testing.T) {
	// [2,3] @ [3,2] = [2,2]
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := FromFloat32([]float32{7, 8, 9, 10, 11, 12}, []int{3, 2})
	c := a.MatMul(b)
	got := c.Data()
	// [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	// [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	want := []float32{58, 64, 139, 154}
	for i := range got {
		if !approx(got[i], want[i], 1e-4) {
			t.Fatalf("c[%d]=%v want %v", i, got[i], want[i])
		}
	}
	if c.Shape()[0] != 2 || c.Shape()[1] != 2 {
		t.Fatalf("shape=%v want [2 2]", c.Shape())
	}
}

func TestMatMulLarger(t *testing.T) {
	// Identity: [4,4] @ I = [4,4]
	data := make([]float32, 16)
	for i := range data {
		data[i] = float32(i + 1)
	}
	a := FromFloat32(data, []int{4, 4})
	eye := FromFloat32([]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, []int{4, 4})
	c := a.MatMul(eye)
	got := c.Data()
	for i := range got {
		if !approx(got[i], data[i], 1e-4) {
			t.Fatalf("[%d]=%v want %v", i, got[i], data[i])
		}
	}
}

func TestPermute(t *testing.T) {
	// [2,3] → [3,2]
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := a.Permute([]int{1, 0})
	got := b.Data()
	// Transpose of [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]
	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func BenchmarkMatMul(b *testing.B) {
	m, k, n := 64, 384, 384
	a := Rand([]int{m, k})
	x := Rand([]int{k, n})
	a.Realize()
	x.Realize()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(x).Data()
	}
}

func TestSoftmax(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3}, []int{1, 3})
	got := a.Softmax().Data()
	// softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
	sum := float32(0)
	for _, v := range got {
		sum += v
		if v < 0 || v > 1 {
			t.Fatalf("softmax value out of range: %v", v)
		}
	}
	if !approx(sum, 1.0, 1e-5) {
		t.Fatalf("softmax sum=%v want 1.0", sum)
	}
	if got[2] < got[1] || got[1] < got[0] {
		t.Fatalf("softmax not monotonic: %v", got)
	}
}

func TestLayerNorm(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	gamma := FromFloat32([]float32{1, 1, 1}, []int{3})
	beta := FromFloat32([]float32{0, 0, 0}, []int{3})
	got := a.LayerNorm(gamma, beta, 1e-5).Data()
	// Each row normalized to ~0 mean, ~1 std
	for i := 0; i < 2; i++ {
		row := got[i*3 : i*3+3]
		mean := (row[0] + row[1] + row[2]) / 3
		if !approx(mean, 0, 0.01) {
			t.Fatalf("row %d mean=%v want ~0", i, mean)
		}
	}
}

func TestGELU(t *testing.T) {
	a := FromFloat32([]float32{-1, 0, 1, 2}, []int{4})
	got := a.GELU().Data()
	if got[1] != 0 {
		t.Fatalf("gelu(0)=%v want 0", got[1])
	}
	if !approx(got[2], 0.841, 0.01) {
		t.Fatalf("gelu(1)=%v want ~0.841", got[2])
	}
}

func TestBroadcastAdd(t *testing.T) {
	// [2,3] + [3] → [2,3]
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := FromFloat32([]float32{10, 20, 30}, []int{3})
	c := a.Add(b)
	got := c.Data()
	want := []float32{11, 22, 33, 14, 25, 36}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestLinear(t *testing.T) {
	// X=[2,3], W=[4,3], bias=[4] → Y=[2,4]
	x := FromFloat32([]float32{1, 0, 0, 0, 1, 0}, []int{2, 3})
	w := FromFloat32([]float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1,
	}, []int{4, 3})
	bias := FromFloat32([]float32{0, 0, 0, 0}, []int{4})
	y := x.Linear(w, bias)
	got := y.Data()
	// Row 0: [1,0,0] @ W^T = [1,0,0,1]
	// Row 1: [0,1,0] @ W^T = [0,1,0,1]
	want := []float32{1, 0, 0, 1, 0, 1, 0, 1}
	for i := range got {
		if !approx(got[i], want[i], 1e-5) {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestTransformerBlock(t *testing.T) {
	// Minimal smoke test: build a fake transformer block
	seqLen, hidden := 4, 8

	// Input
	x := Rand([]int{seqLen, hidden})
	x.Realize()

	// QKV projection
	wQKV := Rand([]int{hidden * 3, hidden})
	wQKV.Realize()
	qkv := x.MatMulTransposed(wQKV)

	// Split Q, K, V (manual for now)
	qkvData := qkv.Data()
	q := FromFloat32(qkvData[:seqLen*hidden], []int{seqLen, hidden})
	k := FromFloat32(qkvData[seqLen*hidden:seqLen*hidden*2], []int{seqLen, hidden})
	v := FromFloat32(qkvData[seqLen*hidden*2:], []int{seqLen, hidden})

	// Attention scores: Q @ K^T
	scores := q.MatMulTransposed(k)

	// Softmax
	probs := scores.Softmax()
	probsData := probs.Data()

	// Check softmax rows sum to 1
	for i := 0; i < seqLen; i++ {
		sum := float32(0)
		for j := 0; j < seqLen; j++ {
			sum += probsData[i*seqLen+j]
		}
		if !approx(sum, 1.0, 1e-4) {
			t.Fatalf("attention row %d sum=%v", i, sum)
		}
	}

	// Context: probs @ V
	ctx := probs.MatMul(v)
	if ctx.Shape()[0] != seqLen || ctx.Shape()[1] != hidden {
		t.Fatalf("context shape=%v want [%d,%d]", ctx.Shape(), seqLen, hidden)
	}
}

func BenchmarkFusedChain5(b *testing.B) {
	// 5-op chain: ((((a+b)*a)-b)+a) — fused into single pass
	n := 1024 * 1024
	x := Rand([]int{n})
	y := Rand([]int{n})
	x.Realize()
	y.Realize()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z := x.Add(y).Mul(x).Sub(y).Add(x)
		z.Realize()
	}
}

func TestFusionCorrectness(t *testing.T) {
	// Verify fused result matches unfused for a chain
	a := FromFloat32([]float32{1, 2, 3, 4}, []int{4})
	b := FromFloat32([]float32{5, 6, 7, 8}, []int{4})

	// (a + b) * a = [6*1, 8*2, 10*3, 12*4] = [6, 16, 30, 48]
	got := a.Add(b).Mul(a).Data()
	want := []float32{6, 16, 30, 48}
	for i := range got {
		if !approx(got[i], want[i], 1e-6) {
			t.Fatalf("[%d]=%v want %v", i, got[i], want[i])
		}
	}

	// Longer: ((a+b)*a - b) + a = [6-5+1, 16-6+2, 30-7+3, 48-8+4] = [2, 12, 26, 44]
	got2 := a.Add(b).Mul(a).Sub(b).Add(a).Data()
	want2 := []float32{2, 12, 26, 44}
	for i := range got2 {
		if !approx(got2[i], want2[i], 1e-6) {
			t.Fatalf("[%d]=%v want %v", i, got2[i], want2[i])
		}
	}
}

func TestShapeValidationRejectsMalformedInputs(t *testing.T) {
	assertPanics(t, func() { _ = NewShape([]int{-1, 2}) })
	assertPanics(t, func() { _ = Zeros([]int{-1}) })
	assertPanics(t, func() { _ = Ones([]int{int(^uint(0) >> 1), 2}) })
	s := NewShape([]int{2, 3})
	assertPanics(t, func() { _ = s.Permute([]int{0}) })
	assertPanics(t, func() { _ = s.Permute([]int{0, 0}) })
	assertPanics(t, func() { _ = s.Permute([]int{0, 2}) })
	assertPanics(t, func() { _ = s.Expand([]int{2}) })
	assertPanics(t, func() { _ = s.Expand([]int{2, -3}) })
	maxInt := int(^uint(0) >> 1)
	badShape := Shape{Dims: []int{maxInt/2 + 1, 3}, Strides: []int{3, 1}}
	if got := badShape.Numel(); got != 0 {
		t.Fatalf("malformed Numel=%d, want 0", got)
	}
	if _, err := badShape.Reshape([]int{1}); err == nil {
		t.Fatal("Reshape accepted malformed source shape")
	}
	big := maxInt/2 + 1
	_, _, _, err := broadcast(NewShape([]int{big, 1}), NewShape([]int{1, 3}))
	if err == nil {
		t.Fatal("broadcast accepted overflowing output shape")
	}
}

func assertPanics(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	fn()
}

func TestReduceValidationRejectsMalformedAxes(t *testing.T) {
	a := FromFloat32([]float32{1, 2, 3, 4}, []int{2, 2})
	assertPanics(t, func() { _ = a.Sum(-1) })
	assertPanics(t, func() { _ = a.Sum(2) })
	assertPanics(t, func() { _ = a.Sum(0, 0) })
	assertPanics(t, func() { _ = a.ReduceMax(3) })
}

func TestTensorNilOperations(t *testing.T) {
	var tns *Tensor
	if got := tns.Data(); got != nil {
		t.Fatalf("nil Data=%v, want nil", got)
	}
	assertPanics(t, func() { tns.Realize() })
	assertPanics(t, func() { tns.Add(Ones([]int{1})) })
	assertPanics(t, func() { Ones([]int{1}).Add(nil) })
	assertPanics(t, func() { tns.Neg() })
}

func TestUnsafeSliceHelpersEmptyInputs(t *testing.T) {
	assertPanics(t, func() { _ = byteSliceToFloat32([]byte{1, 2, 3}) })
	if got := byteSliceToFloat32(nil); got != nil {
		t.Fatalf("byteSliceToFloat32(nil)=%v, want nil", got)
	}
	if got := float32ToByteSlice(nil); got != nil {
		t.Fatalf("float32ToByteSlice(nil)=%v, want nil", got)
	}
	var b *Buffer
	if got := b.Float32Data(); got != nil {
		t.Fatalf("nil Buffer Float32Data=%v, want nil", got)
	}
	assertPanics(t, func() { _ = (&Buffer{Data: []byte{1, 2, 3, 4}, DType: Float32, Length: -1}).Float32Data() })
	assertPanics(t, func() { _ = (&Buffer{Data: []byte{1, 2, 3}, DType: Float32, Length: 1}).Float32Data() })
	assertPanics(t, func() { _ = (&Buffer{Data: []byte{1, 2, 3, 4}, DType: Float32, Length: 2}).Float32Data() })
	z := Zeros([]int{0})
	if got := z.Data(); got != nil {
		t.Fatalf("zero-size tensor Data=%v, want nil", got)
	}
}

func TestRealizeValidationRejectsMalformedUOps(t *testing.T) {
	assertPanics(t, func() { _ = realize(nil, NewShape([]int{1})) })
	assertPanics(t, func() { _ = realize(&UOp{Op: OpNeg, DType: Float32, Src: []*UOp{nil}}, NewShape([]int{1})) })
	assertPanics(t, func() { _ = unaryEval(nil, 1, func(v float32) float32 { return v }) })
	assertPanics(t, func() { _ = binaryEval(nil, nil, 1, func(a, b float32) float32 { return a + b }) })
	assertPanics(t, func() {
		_ = reduceEval(nil, NewShape([]int{1}), []int{0}, func(a, b float32) float32 { return a + b }, 0)
	})
	assertPanics(t, func() { _ = guessInputShape(&UOp{}) })
}

func TestPooledAllocValidation(t *testing.T) {
	assertPanics(t, func() { _ = pooledAlloc(Float32, -1) })
	assertPanics(t, func() { _ = pooledAlloc(DType{Name: "bad", Bits: 0}, 1) })
	assertPanics(t, func() { _ = pooledAlloc(Float32, int(^uint(0)>>1)) })
}

func TestFusionValidationRejectsMalformedKernels(t *testing.T) {
	if got := tryFuse(nil, NewShape([]int{1})); got != nil {
		t.Fatalf("tryFuse(nil)=%v, want nil", got)
	}
	assertPanics(t, func() { (&fusedKernel{}).execute() })
	assertPanics(t, func() {
		(&fusedKernel{n: 2, ops: []fusedOp{{isLeaf: true, bufIdx: 0}}, bufs: []*Buffer{{Data: float32ToByteSlice([]float32{1}), DType: Float32, Length: 1}}}).execute()
	})
}

func TestEmbeddingValidation(t *testing.T) {
	assertPanics(t, func() { _ = Embedding(nil, []int{0}) })
	w := FromFloat32([]float32{1, 2, 3, 4}, []int{2, 2})
	assertPanics(t, func() { _ = Embedding(w, []int{-1}) })
	assertPanics(t, func() { _ = Embedding(w, []int{2}) })
	got := Embedding(w, nil)
	if shape := got.Shape(); len(shape) != 2 || shape[0] != 0 || shape[1] != 2 {
		t.Fatalf("empty embedding shape=%v", shape)
	}
}

func TestMatMulAndLinearValidation(t *testing.T) {
	assertPanics(t, func() { _ = (*Tensor)(nil).MatMul(Ones([]int{1, 1})) })
	assertPanics(t, func() { _ = Ones([]int{1, 1}).MatMul(nil) })
	assertPanics(t, func() { _ = Ones([]int{1}).MatMulTransposed(Ones([]int{1, 1})) })
	assertPanics(t, func() { _ = Ones([]int{1, 2}).Linear(Ones([]int{3, 2}), Ones([]int{2})) })
	assertPanics(t, func() { _ = Ones([]int{1, 2}).LinearPreT(Ones([]int{2, 3}), Ones([]int{2})) })
	got := Ones([]int{0, 2}).MatMul(Ones([]int{2, 3}))
	if shape := got.Shape(); len(shape) != 2 || shape[0] != 0 || shape[1] != 3 {
		t.Fatalf("zero-row matmul shape=%v", shape)
	}
}

func TestNNHelperValidation(t *testing.T) {
	assertPanics(t, func() { _ = (*Tensor)(nil).Softmax() })
	assertPanics(t, func() { _ = (*Tensor)(nil).LayerNorm(nil, nil, 1e-5) })
	assertPanics(t, func() { _ = (*Tensor)(nil).GELU() })
	x := Ones([]int{2, 3})
	assertPanics(t, func() { _ = x.LayerNorm(Ones([]int{2}), Ones([]int{3}), 1e-5) })
	assertPanics(t, func() { _ = x.LayerNorm(Ones([]int{3}), Ones([]int{2}), 1e-5) })
	assertPanics(t, func() { _ = x.LayerNorm(Ones([]int{3}), nil, 1e-5) })
	z := Zeros([]int{2, 0})
	if got := z.Softmax(); got.Numel() != 0 {
		t.Fatalf("zero-width softmax numel=%d", got.Numel())
	}
	if got := z.LayerNorm(nil, nil, 1e-5); got.Numel() != 0 {
		t.Fatalf("zero-width layernorm numel=%d", got.Numel())
	}
}

func TestModuleConstructorValidation(t *testing.T) {
	assertPanics(t, func() { _ = NewLinear(0, 1) })
	assertPanics(t, func() { _ = NewLinear(1, -1) })
	assertPanics(t, func() { _ = NewLayerNorm(-1) })
	assertPanics(t, func() { _ = NewEmbedding(-1, 1) })
	assertPanics(t, func() { _ = NewEmbedding(1, -1) })
	assertPanics(t, func() { _ = (*LinearModule)(nil).Forward(Ones([]int{1, 1})) })
	assertPanics(t, func() { _ = (*LayerNormModule)(nil).Forward(Ones([]int{1, 1})) })
	assertPanics(t, func() { _ = (*EmbeddingModule)(nil).Forward([]int{0}) })
}

func TestTensorPropertiesNilSafe(t *testing.T) {
	var x *Tensor
	if x.Shape() != nil || x.Ndim() != 0 || x.Numel() != 0 || x.DType().Name != "" {
		t.Fatalf("nil tensor properties not zero-valued")
	}
}

func TestShapeIsContiguousRejectsMalformedShape(t *testing.T) {
	if (Shape{Dims: []int{2, 2}, Strides: []int{2}}).IsContiguous() {
		t.Fatal("IsContiguous accepted mismatched strides")
	}
	if (Shape{Dims: []int{-1}, Strides: []int{1}}).IsContiguous() {
		t.Fatal("IsContiguous accepted invalid dims")
	}
	maxInt := int(^uint(0) >> 1)
	if (Shape{Dims: []int{maxInt/2 + 1, 3}, Strides: []int{3, 1}}).IsContiguous() {
		t.Fatal("IsContiguous accepted overflowing dims")
	}
}

func TestTensorConvenienceOpsValidateInputs(t *testing.T) {
	var nilTensor *Tensor
	assertPanics(t, func() { nilTensor.Transpose2D() })
	assertPanics(t, func() { nilTensor.Clip(0, 1) })
	assertPanics(t, func() { nilTensor.ReLU() })
	assertPanics(t, func() { nilTensor.Sigmoid() })
	assertPanics(t, func() { Where(nil, Ones([]int{1}), Ones([]int{1})) })
	assertPanics(t, func() { Where(Ones([]int{2}), Ones([]int{2}), Ones([]int{3})) })

	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	tr := a.Transpose2D().Data()
	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range want {
		if tr[i] != want[i] {
			t.Fatalf("Transpose2D[%d]=%v want %v", i, tr[i], want[i])
		}
	}
}
