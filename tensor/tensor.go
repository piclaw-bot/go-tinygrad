package tensor

import (
	"math"
	"math/rand"
)

// Tensor is the user-facing type. It wraps a UOp graph node.
// All operations are lazy until Realize() is called.
type Tensor struct {
	uop   *UOp
	shape Shape
}

// --- Constructors ---

// FromFloat32 creates a tensor from a float32 slice.
func FromFloat32(data []float32, shape []int) *Tensor {
	n := shapeSize(shape)
	if n < 0 || n != len(data) {
		panic("shape mismatch")
	}
	u := BufferOp(Float32, shape)
	u.buf = &Buffer{
		Data:   float32ToByteSlice(append([]float32(nil), data...)),
		DType:  Float32,
		Length: len(data),
	}
	return &Tensor{uop: u, shape: NewShape(shape)}
}

// Zeros creates a zero-filled tensor.
func Zeros(shape []int) *Tensor {
	n := shapeSize(shape)
	if n < 0 {
		panic("invalid shape")
	}
	return FromFloat32(make([]float32, n), shape)
}

// Ones creates a ones-filled tensor.
func Ones(shape []int) *Tensor {
	n := shapeSize(shape)
	if n < 0 {
		panic("invalid shape")
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = 1
	}
	return FromFloat32(data, shape)
}

// Rand creates a tensor with uniform random values in [0, 1).
func Rand(shape []int) *Tensor {
	n := shapeSize(shape)
	if n < 0 {
		panic("invalid shape")
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()
	}
	return FromFloat32(data, shape)
}

// Full creates a tensor filled with a constant value.
func Full(shape []int, val float32) *Tensor {
	n := shapeSize(shape)
	if n < 0 {
		panic("invalid shape")
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = val
	}
	return FromFloat32(data, shape)
}

// --- Properties ---

func (t *Tensor) Shape() []int {
	if t == nil {
		return nil
	}
	return t.shape.Dims
}
func (t *Tensor) DType() DType {
	if t == nil || t.uop == nil {
		return DType{}
	}
	return t.uop.DType
}
func (t *Tensor) Ndim() int {
	if t == nil {
		return 0
	}
	return t.shape.Ndim()
}
func (t *Tensor) Numel() int {
	if t == nil {
		return 0
	}
	return t.shape.Numel()
}

// --- Lazy binary ops ---

func (t *Tensor) Add(other *Tensor) *Tensor {
	return t.binaryOp(OpAdd, other)
}

func (t *Tensor) Sub(other *Tensor) *Tensor {
	return t.binaryOp(OpSub, other)
}

func (t *Tensor) Mul(other *Tensor) *Tensor {
	return t.binaryOp(OpMul, other)
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	return t.binaryOp(OpDiv, other)
}

func (t *Tensor) Max(other *Tensor) *Tensor {
	return t.binaryOp(OpMax, other)
}

// --- Lazy unary ops ---

func (t *Tensor) Neg() *Tensor   { return t.unaryOp(OpNeg) }
func (t *Tensor) Sqrt() *Tensor  { return t.unaryOp(OpSqrt) }
func (t *Tensor) Exp2() *Tensor  { return t.unaryOp(OpExp2) }
func (t *Tensor) Log2() *Tensor  { return t.unaryOp(OpLog2) }
func (t *Tensor) Recip() *Tensor { return t.unaryOp(OpReciprocal) }

// --- Lazy reduce ops ---

func (t *Tensor) Sum(axes ...int) *Tensor {
	return t.reduceOp(OpReduceSum, axes)
}

func (t *Tensor) ReduceMax(axes ...int) *Tensor {
	return t.reduceOp(OpReduceMax, axes)
}

// --- Movement ops ---

func (t *Tensor) Reshape(newShape []int) *Tensor {
	s, err := t.shape.Reshape(newShape)
	if err != nil {
		panic(err)
	}
	u := newUOp(OpReshape, t.uop.DType, []*UOp{t.uop}, cloneShape(newShape))
	return &Tensor{uop: u, shape: s}
}

func (t *Tensor) Permute(order []int) *Tensor {
	s := t.shape.Permute(order)
	u := newUOp(OpPermute, t.uop.DType, []*UOp{t.uop}, cloneShape(order))
	return &Tensor{uop: u, shape: s}
}

// --- Realize ---

// Realize executes the computation graph and returns the tensor with data.
func (t *Tensor) Realize() *Tensor {
	if t == nil || t.uop == nil {
		panic("realize: nil tensor")
	}
	if t.uop.buf != nil {
		return t // already realized
	}
	t.uop.buf = realize(t.uop, t.shape)
	return t
}

// Data returns the realized float32 data. Panics if not realized.
func (t *Tensor) Data() []float32 {
	if t == nil || t.uop == nil {
		return nil
	}
	if t.uop.buf == nil {
		t.Realize()
	}
	return t.uop.buf.Float32Data()
}

// --- Internal helpers ---

func (t *Tensor) binaryOp(op Ops, other *Tensor) *Tensor {
	if t == nil || other == nil || t.uop == nil || other.uop == nil {
		panic("binary op: nil tensor")
	}
	if len(t.shape.Dims) == len(other.shape.Dims) {
		match := true
		for i := range t.shape.Dims {
			if t.shape.Dims[i] != other.shape.Dims[i] {
				match = false
				break
			}
		}
		if match {
			u := newUOp(op, t.uop.DType, []*UOp{t.uop, other.uop}, nil)
			return &Tensor{uop: u, shape: t.shape}
		}
	}
	outDims, _, _, err := broadcast(t.shape, other.shape)
	if err != nil {
		panic(err)
	}
	arg := BroadcastArg{outDims, cloneShape(t.shape.Dims), cloneShape(other.shape.Dims)}
	// Don't intern broadcast ops (Arg contains shapes needed for realize)
	u := &UOp{Op: op, DType: t.uop.DType, Src: []*UOp{t.uop, other.uop}, Arg: arg}
	return &Tensor{uop: u, shape: NewShape(outDims)}
}

func (t *Tensor) unaryOp(op Ops) *Tensor {
	if t == nil || t.uop == nil {
		panic("unary op: nil tensor")
	}
	u := newUOp(op, t.uop.DType, []*UOp{t.uop}, nil)
	return &Tensor{uop: u, shape: t.shape}
}

func (t *Tensor) reduceOp(op Ops, axes []int) *Tensor {
	if t == nil {
		panic("reduce: nil tensor")
	}
	ndim := len(t.shape.Dims)
	seen := make([]bool, ndim)
	newDims := make([]int, ndim)
	copy(newDims, t.shape.Dims)
	for _, ax := range axes {
		if ax < 0 || ax >= ndim || seen[ax] {
			panic("reduce: invalid axes")
		}
		seen[ax] = true
		newDims[ax] = 1
	}
	u := newUOp(op, t.uop.DType, []*UOp{t.uop}, cloneShape(axes))
	return &Tensor{uop: u, shape: NewShape(newDims)}
}

// Transpose2D returns a transposed copy of a 2D tensor.
// [M, N] → [N, M]. Eagerly realized.
func (t *Tensor) Transpose2D() *Tensor {
	if t == nil || t.uop == nil {
		panic("Transpose2D: nil tensor")
	}
	t.Realize()
	dims := t.Shape()
	if len(dims) != 2 {
		panic("Transpose2D requires 2D tensor")
	}
	m, n := dims[0], dims[1]
	data := t.Data()
	total := shapeSize(dims)
	if total < 0 || len(data) < total {
		panic("Transpose2D: invalid backing data")
	}
	out := make([]float32, total)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out[j*m+i] = data[i*n+j]
		}
	}
	return FromFloat32(out, []int{n, m})
}

// Where selects elements: out[i] = cond[i] ? x[i] : y[i]
func Where(cond, x, y *Tensor) *Tensor {
	if cond == nil || x == nil || y == nil || cond.uop == nil || x.uop == nil || y.uop == nil {
		panic("where: nil tensor")
	}
	outDims, _, _, err := broadcast(x.shape, y.shape)
	if err != nil {
		panic(err)
	}
	condDims, _, _, err := broadcast(cond.shape, NewShape(outDims))
	if err != nil {
		panic(err)
	}
	u := newUOp(OpWhere, x.uop.DType, []*UOp{cond.uop, x.uop, y.uop}, nil)
	return &Tensor{uop: u, shape: NewShape(condDims)}
}

// CmpLt returns a boolean tensor: out[i] = (a[i] < b[i])
func (t *Tensor) CmpLt(other *Tensor) *Tensor {
	return t.binaryOp(OpCmpLt, other)
}

// CmpEq returns a boolean tensor: out[i] = (a[i] == b[i])
func (t *Tensor) CmpEq(other *Tensor) *Tensor {
	return t.binaryOp(OpCmpEq, other)
}

// Clip clamps values to [min, max].
func (t *Tensor) Clip(min, max float32) *Tensor {
	if t == nil || t.uop == nil {
		panic("clip: nil tensor")
	}
	lo := Full(t.Shape(), min)
	hi := Full(t.Shape(), max)
	return t.Max(lo).Add(hi.Neg()).Neg().Add(hi)
}

// ReLU returns max(0, x).
func (t *Tensor) ReLU() *Tensor {
	if t == nil || t.uop == nil {
		panic("relu: nil tensor")
	}
	zero := Zeros(t.Shape())
	return t.Max(zero)
}

// Sigmoid returns 1 / (1 + exp(-x)).
func (t *Tensor) Sigmoid() *Tensor {
	if t == nil || t.uop == nil {
		panic("sigmoid: nil tensor")
	}
	// sigmoid(x) = 1 / (1 + exp2(-x / ln2))
	// Approximation via exp2
	t.Realize()
	data := t.Data()
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = 1.0 / (1.0 + float32(math.Exp(float64(-v))))
	}
	return FromFloat32(out, t.Shape())
}
