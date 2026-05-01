package tensor

import "math"

// realize executes a UOp graph eagerly and returns the result buffer.
// This is the simple interpreter — no fusion, no scheduling.
// Will be replaced by a proper scheduler+fuser later.
func realize(u *UOp, shape Shape) *Buffer {
	// Recursively realize inputs
	for _, src := range u.Src {
		if src.buf == nil && src.Op != OpConst {
			srcShape := shape // TODO: track shapes properly per node
			src.buf = realize(src, srcShape)
		}
	}

	n := shape.Numel()
	switch u.Op {
	case OpBuffer:
		if u.buf != nil {
			return u.buf
		}
		return allocBuffer(Float32, n)

	case OpConst:
		val := float32(u.Arg.(float64))
		buf := allocBuffer(u.DType, n)
		data := buf.Float32Data()
		for i := range data {
			data[i] = val
		}
		return buf

	case OpAdd:
		return binaryEval(u.Src[0].buf, u.Src[1].buf, n, func(a, b float32) float32 { return a + b })
	case OpSub:
		return binaryEval(u.Src[0].buf, u.Src[1].buf, n, func(a, b float32) float32 { return a - b })
	case OpMul:
		return binaryEval(u.Src[0].buf, u.Src[1].buf, n, func(a, b float32) float32 { return a * b })
	case OpDiv:
		return binaryEval(u.Src[0].buf, u.Src[1].buf, n, func(a, b float32) float32 { return a / b })
	case OpMax:
		return binaryEval(u.Src[0].buf, u.Src[1].buf, n, func(a, b float32) float32 {
			if a > b {
				return a
			}
			return b
		})

	case OpNeg:
		return unaryEval(u.Src[0].buf, n, func(a float32) float32 { return -a })
	case OpSqrt:
		return unaryEval(u.Src[0].buf, n, func(a float32) float32 { return float32(math.Sqrt(float64(a))) })
	case OpExp2:
		return unaryEval(u.Src[0].buf, n, func(a float32) float32 { return float32(math.Exp2(float64(a))) })
	case OpLog2:
		return unaryEval(u.Src[0].buf, n, func(a float32) float32 { return float32(math.Log2(float64(a))) })
	case OpReciprocal:
		return unaryEval(u.Src[0].buf, n, func(a float32) float32 { return 1.0 / a })

	case OpReduceSum:
		srcShape := guessInputShape(u)
		return reduceEval(u.Src[0].buf, srcShape, u.Arg.([]int), func(acc, v float32) float32 { return acc + v }, 0)
	case OpReduceMax:
		srcShape := guessInputShape(u)
		return reduceEval(u.Src[0].buf, srcShape, u.Arg.([]int), func(acc, v float32) float32 {
			if v > acc {
				return v
			}
			return acc
		}, -math.MaxFloat32)

	case OpReshape:
		// Reshape is a view — just pass through the buffer
		return u.Src[0].buf

	default:
		panic("realize: unimplemented op " + u.Op.String())
	}
}

func allocBuffer(dtype DType, n int) *Buffer {
	return &Buffer{
		Data:   make([]byte, n*dtype.ByteSize()),
		DType:  dtype,
		Length: n,
	}
}

func unaryEval(src *Buffer, n int, f func(float32) float32) *Buffer {
	out := allocBuffer(src.DType, n)
	a := src.Float32Data()
	o := out.Float32Data()
	for i := 0; i < n; i++ {
		o[i] = f(a[i])
	}
	return out
}

func binaryEval(lhs, rhs *Buffer, n int, f func(float32, float32) float32) *Buffer {
	out := allocBuffer(lhs.DType, n)
	a := lhs.Float32Data()
	b := rhs.Float32Data()
	o := out.Float32Data()
	for i := 0; i < n; i++ {
		o[i] = f(a[i], b[i])
	}
	return out
}

func reduceEval(src *Buffer, shape Shape, axes []int, f func(float32, float32) float32, init float32) *Buffer {
	data := src.Float32Data()
	outDims := make([]int, len(shape.Dims))
	copy(outDims, shape.Dims)
	for _, ax := range axes {
		outDims[ax] = 1
	}
	outShape := NewShape(outDims)
	outN := outShape.Numel()
	out := allocBuffer(src.DType, outN)
	result := out.Float32Data()
	for i := range result {
		result[i] = init
	}

	ndim := len(shape.Dims)
	idx := make([]int, ndim)
	for i := 0; i < shape.Numel(); i++ {
		// Compute flat output index
		outIdx := 0
		for d := 0; d < ndim; d++ {
			dimIdx := idx[d]
			isReduced := false
			for _, ax := range axes {
				if ax == d {
					isReduced = true
					break
				}
			}
			if !isReduced {
				outIdx += dimIdx * outShape.Strides[d]
			}
		}
		result[outIdx] = f(result[outIdx], data[i])

		for d := ndim - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape.Dims[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

// guessInputShape recovers the input shape from a reduce UOp.
// The source tensor's shape has the unreduced dimensions.
func guessInputShape(u *UOp) Shape {
	if u.Src[0].Op == OpBuffer {
		return NewShape(u.Src[0].Arg.([]int))
	}
	// For chained ops, walk to the buffer
	node := u.Src[0]
	for node.Op != OpBuffer && len(node.Src) > 0 {
		node = node.Src[0]
	}
	if node.Op == OpBuffer {
		return NewShape(node.Arg.([]int))
	}
	// Fallback: reconstruct from buffer length
	if u.Src[0].buf != nil {
		n := u.Src[0].buf.Length
		return NewShape([]int{n})
	}
	panic("cannot determine input shape for reduce")
}
