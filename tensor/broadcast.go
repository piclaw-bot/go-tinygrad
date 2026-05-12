package tensor

import "fmt"

// broadcast computes the broadcast shape of two tensors and returns
// expanded shapes for both inputs.
func broadcast(a, b Shape) (outDims []int, aExp, bExp Shape, err error) {
	if shapeSize(a.Dims) < 0 || shapeSize(b.Dims) < 0 {
		return nil, Shape{}, Shape{}, fmt.Errorf("broadcast: invalid shapes %v and %v", a.Dims, b.Dims)
	}
	na, nb := a.Ndim(), b.Ndim()
	ndim := na
	if nb > ndim {
		ndim = nb
	}

	// Pad shorter shape with 1s on the left
	aDims := make([]int, ndim)
	bDims := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		ai := 1
		if idx := i - (ndim - na); idx >= 0 {
			ai = a.Dims[idx]
		}
		bi := 1
		if idx := i - (ndim - nb); idx >= 0 {
			bi = b.Dims[idx]
		}
		aDims[i] = ai
		bDims[i] = bi
	}

	outDims = make([]int, ndim)
	maxInt := int(^uint(0) >> 1)
	numel := 1
	for i := 0; i < ndim; i++ {
		if aDims[i] < 0 || bDims[i] < 0 {
			return nil, Shape{}, Shape{}, fmt.Errorf("broadcast: invalid dims %d: %d vs %d", i, aDims[i], bDims[i])
		}
		if aDims[i] == bDims[i] {
			outDims[i] = aDims[i]
		} else if aDims[i] == 1 {
			outDims[i] = bDims[i]
		} else if bDims[i] == 1 {
			outDims[i] = aDims[i]
		} else {
			return nil, Shape{}, Shape{}, fmt.Errorf("broadcast: incompatible dims %d: %d vs %d", i, aDims[i], bDims[i])
		}
		if outDims[i] != 0 && numel > maxInt/outDims[i] {
			return nil, Shape{}, Shape{}, fmt.Errorf("broadcast: output shape overflows: %v", outDims)
		}
		numel *= outDims[i]
	}

	aExp = NewShape(aDims).Expand(outDims)
	bExp = NewShape(bDims).Expand(outDims)
	return outDims, aExp, bExp, nil
}
