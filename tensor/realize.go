package tensor

import "math"

func realize(u *UOp, shape Shape) *Buffer {
	// Try elementwise fusion first
	if k := tryFuse(u, shape); k != nil {
		return k.execute()
	}

	for _, src := range u.Src {
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
			if a > b { return a }; return b
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
			if v > acc { return v }; return acc
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
				if outIdx[d] < shape.Dims[d] { break }
				outIdx[d] = 0
			}
		}
		return outBuf

	default:
		panic("realize: unimplemented op " + u.Op.String())
	}
}

func allocBuffer(dtype DType, n int) *Buffer {
	return &Buffer{Data: make([]byte, n*dtype.ByteSize()), DType: dtype, Length: n}
}

func unaryEval(src *Buffer, n int, f func(float32) float32) *Buffer {
	out := allocBuffer(src.DType, n)
	a, o := src.Float32Data(), out.Float32Data()
	for i := 0; i < n; i++ { o[i] = f(a[i]) }
	return out
}

func binaryEval(lhs, rhs *Buffer, n int, f func(float32, float32) float32) *Buffer {
	out := allocBuffer(lhs.DType, n)
	a, b, o := lhs.Float32Data(), rhs.Float32Data(), out.Float32Data()
	for i := 0; i < n; i++ { o[i] = f(a[i], b[i]) }
	return out
}

func binaryBroadcastEval(u *UOp, outShape Shape, f func(float32, float32) float32) *Buffer {
	lhs, rhs := u.Src[0].buf, u.Src[1].buf
	n := outShape.Numel()

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
			if idx[d] < outShape.Dims[d] { break }
			idx[d] = 0
		}
	}
	return out
}

func reduceEval(src *Buffer, shape Shape, axes []int, f func(float32, float32) float32, init float32) *Buffer {
	data := src.Float32Data()
	outDims := make([]int, len(shape.Dims))
	copy(outDims, shape.Dims)
	for _, ax := range axes { outDims[ax] = 1 }
	outShape := NewShape(outDims)
	out := allocBuffer(src.DType, outShape.Numel())
	result := out.Float32Data()
	for i := range result { result[i] = init }

	ndim := len(shape.Dims)
	idx := make([]int, ndim)
	for i := 0; i < shape.Numel(); i++ {
		outIdx := 0
		for d := 0; d < ndim; d++ {
			isReduced := false
			for _, ax := range axes { if ax == d { isReduced = true; break } }
			if !isReduced { outIdx += idx[d] * outShape.Strides[d] }
		}
		result[outIdx] = f(result[outIdx], data[i])
		for d := ndim - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape.Dims[d] { break }
			idx[d] = 0
		}
	}
	return out
}

func guessInputShape(u *UOp) Shape {
	node := u.Src[0]
	for node.Op != OpBuffer && len(node.Src) > 0 { node = node.Src[0] }
	if node.Op == OpBuffer && node.Arg != nil { return NewShape(node.Arg.([]int)) }
	if u.Src[0].buf != nil { return NewShape([]int{u.Src[0].buf.Length}) }
	panic("cannot determine input shape")
}
