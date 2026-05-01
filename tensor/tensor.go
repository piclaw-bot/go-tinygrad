package tensor

import (
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
	if shapeSize(shape) != len(data) {
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
	return FromFloat32(make([]float32, n), shape)
}

// Ones creates a ones-filled tensor.
func Ones(shape []int) *Tensor {
	n := shapeSize(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = 1
	}
	return FromFloat32(data, shape)
}

// Rand creates a tensor with uniform random values in [0, 1).
func Rand(shape []int) *Tensor {
	n := shapeSize(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()
	}
	return FromFloat32(data, shape)
}

// Full creates a tensor filled with a constant value.
func Full(shape []int, val float32) *Tensor {
	n := shapeSize(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = val
	}
	return FromFloat32(data, shape)
}

// --- Properties ---

func (t *Tensor) Shape() []int  { return t.shape.Dims }
func (t *Tensor) DType() DType  { return t.uop.DType }
func (t *Tensor) Ndim() int     { return t.shape.Ndim() }
func (t *Tensor) Numel() int    { return t.shape.Numel() }

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

func (t *Tensor) Neg() *Tensor    { return t.unaryOp(OpNeg) }
func (t *Tensor) Sqrt() *Tensor   { return t.unaryOp(OpSqrt) }
func (t *Tensor) Exp2() *Tensor   { return t.unaryOp(OpExp2) }
func (t *Tensor) Log2() *Tensor   { return t.unaryOp(OpLog2) }
func (t *Tensor) Recip() *Tensor  { return t.unaryOp(OpReciprocal) }

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
	if t.uop.buf != nil {
		return t // already realized
	}
	t.uop.buf = realize(t.uop, t.shape)
	return t
}

// Data returns the realized float32 data. Panics if not realized.
func (t *Tensor) Data() []float32 {
	if t.uop.buf == nil {
		t.Realize()
	}
	return t.uop.buf.Float32Data()
}

// --- Internal helpers ---

func (t *Tensor) binaryOp(op Ops, other *Tensor) *Tensor {
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
	u := newUOp(op, t.uop.DType, []*UOp{t.uop}, nil)
	return &Tensor{uop: u, shape: t.shape}
}

func (t *Tensor) reduceOp(op Ops, axes []int) *Tensor {
	newDims := make([]int, len(t.shape.Dims))
	copy(newDims, t.shape.Dims)
	for _, ax := range axes {
		newDims[ax] = 1
	}
	u := newUOp(op, t.uop.DType, []*UOp{t.uop}, cloneShape(axes))
	return &Tensor{uop: u, shape: NewShape(newDims)}
}

// Transpose2D returns a transposed copy of a 2D tensor.
// [M, N] → [N, M]. Eagerly realized.
func (t *Tensor) Transpose2D() *Tensor {
	t.Realize()
	dims := t.Shape()
	if len(dims) != 2 {
		panic("Transpose2D requires 2D tensor")
	}
	m, n := dims[0], dims[1]
	data := t.Data()
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out[j*m+i] = data[i*n+j]
		}
	}
	return FromFloat32(out, []int{n, m})
}
