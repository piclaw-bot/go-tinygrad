package tensor

import "math"

func realize(u *UOp, shape Shape) *Buffer {
	if u == nil {
		panic("realize: nil UOp")
	}
	if shape.Numel() < 0 {
		panic("realize: invalid shape")
	}
	// Try elementwise fusion first
	if k := tryFuse(u, shape); k != nil {
		return k.execute()
	}

	for _, src := range u.Src {
		if src == nil {
			panic("realize: nil source")
		}
		if src.buf == nil && src.Op != OpConst {
			srcShape := shape
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
		for i, data := 0, buf.Float32Data(); i < n; i++ {
			data[i] = val
		}
		return buf

	case OpAdd:
		return binaryBroadcastEval(u, shape, func(a, b float32) float32 { return a + b })
	case OpSub:
		return binaryBroadcastEval(u, shape, func(a, b float32) float32 { return a - b })
	case OpMul:
		return binaryBroadcastEval(u, shape, func(a, b float32) float32 { return a * b })
	case OpDiv:
		return binaryBroadcastEval(u, shape, func(a, b float32) float32 { return a / b })
	case OpMax:
		return binaryBroadcastEval(u, shape, func(a, b float32) float32 {
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
		return u.Src[0].buf

	case OpPermute:
		srcBuf := u.Src[0].buf
		order := u.Arg.([]int)
		srcData := srcBuf.Float32Data()
		srcShape := guessInputShape(u)
		outBuf := allocBuffer(srcBuf.DType, shape.Numel())
		outData := outBuf.Float32Data()
		ndim := len(order)
		outIdx := make([]int, ndim)
		for i := 0; i < shape.Numel(); i++ {
			srcFlat := 0
			for d := 0; d < ndim; d++ {
				srcFlat += outIdx[d] * srcShape.Strides[order[d]]
			}
			outData[i] = srcData[srcFlat]
			for d := ndim - 1; d >= 0; d-- {
				outIdx[d]++
				if outIdx[d] < shape.Dims[d] {
					break
				}
				outIdx[d] = 0
			}
		}
		return outBuf

	default:
		panic("realize: unimplemented op " + u.Op.String())
	}
}

func allocBuffer(dtype DType, n int) *Buffer {
	if n < 0 {
		panic("allocBuffer: negative length")
	}
	return pooledAlloc(dtype, n)
}

func unaryEval(src *Buffer, n int, f func(float32) float32) *Buffer {
	if src == nil || n < 0 || src.Length < n {
		panic("unaryEval: invalid source")
	}
	out := allocBuffer(src.DType, n)
	a, o := src.Float32Data(), out.Float32Data()
	for i := 0; i < n; i++ {
		o[i] = f(a[i])
	}
	return out
}

func binaryEval(lhs, rhs *Buffer, n int, f func(float32, float32) float32) *Buffer {
	if lhs == nil || rhs == nil || n < 0 || lhs.Length < n || rhs.Length < n {
		panic("binaryEval: invalid inputs")
	}
	out := allocBuffer(lhs.DType, n)
	a, b, o := lhs.Float32Data(), rhs.Float32Data(), out.Float32Data()
	for i := 0; i < n; i++ {
		o[i] = f(a[i], b[i])
	}
	return out
}

func binaryBroadcastEval(u *UOp, outShape Shape, f func(float32, float32) float32) *Buffer {
	if u == nil || len(u.Src) < 2 || u.Src[0] == nil || u.Src[1] == nil {
		panic("broadcast binary op without two sources")
	}
	lhs, rhs := u.Src[0].buf, u.Src[1].buf
	n := outShape.Numel()
	if lhs == nil || rhs == nil || n < 0 {
		panic("broadcast binary op with invalid buffers")
	}

	if lhs.Length == n && rhs.Length == n {
		return binaryEval(lhs, rhs, n, f)
	}

	shapes, ok := u.Arg.(BroadcastArg)
	if !ok {
		panic("broadcast binary op without BroadcastArg")
	}
	lhsShape := NewShape(shapes.LhsDims)
	rhsShape := NewShape(shapes.RhsDims)

	a, b := lhs.Float32Data(), rhs.Float32Data()
	out := allocBuffer(lhs.DType, n)
	o := out.Float32Data()
	ndim := outShape.Ndim()
	idx := make([]int, ndim)

	for i := 0; i < n; i++ {
		aIdx, bIdx := 0, 0
		for d := 0; d < ndim; d++ {
			aOff := d - (ndim - lhsShape.Ndim())
			if aOff >= 0 && lhsShape.Dims[aOff] > 1 {
				aIdx += idx[d] * lhsShape.Strides[aOff]
			}
			bOff := d - (ndim - rhsShape.Ndim())
			if bOff >= 0 && rhsShape.Dims[bOff] > 1 {
				bIdx += idx[d] * rhsShape.Strides[bOff]
			}
		}
		o[i] = f(a[aIdx], b[bIdx])
		for d := ndim - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < outShape.Dims[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

func reduceEval(src *Buffer, shape Shape, axes []int, f func(float32, float32) float32, init float32) *Buffer {
	if src == nil || shape.Numel() < 0 || src.Length < shape.Numel() {
		panic("reduceEval: invalid source")
	}
	data := src.Float32Data()
	outDims := make([]int, len(shape.Dims))
	copy(outDims, shape.Dims)
	seen := make([]bool, len(shape.Dims))
	for _, ax := range axes {
		if ax < 0 || ax >= len(shape.Dims) || seen[ax] {
			panic("reduceEval: invalid axes")
		}
		seen[ax] = true
		outDims[ax] = 1
	}
	outShape := NewShape(outDims)
	out := allocBuffer(src.DType, outShape.Numel())
	result := out.Float32Data()
	for i := range result {
		result[i] = init
	}

	ndim := len(shape.Dims)
	idx := make([]int, ndim)
	for i := 0; i < shape.Numel(); i++ {
		outIdx := 0
		for d := 0; d < ndim; d++ {
			isReduced := false
			for _, ax := range axes {
				if ax == d {
					isReduced = true
					break
				}
			}
			if !isReduced {
				outIdx += idx[d] * outShape.Strides[d]
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

func guessInputShape(u *UOp) Shape {
	if u == nil || len(u.Src) == 0 || u.Src[0] == nil {
		panic("cannot determine input shape")
	}
	node := u.Src[0]
	for node.Op != OpBuffer && len(node.Src) > 0 {
		if node.Src[0] == nil {
			panic("cannot determine input shape")
		}
		node = node.Src[0]
	}
	if node.Op == OpBuffer && node.Arg != nil {
		shape, ok := node.Arg.([]int)
		if !ok {
			panic("cannot determine input shape")
		}
		return NewShape(shape)
	}
	if u.Src[0].buf != nil {
		return NewShape([]int{u.Src[0].buf.Length})
	}
	panic("cannot determine input shape")
}
