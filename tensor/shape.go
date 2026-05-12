package tensor

import "fmt"

// Shape represents a tensor's dimensions with strides for views.
type Shape struct {
	Dims    []int
	Strides []int
}

// NewShape creates a contiguous shape.
func NewShape(dims []int) Shape {
	if shapeSize(dims) < 0 {
		panic(fmt.Sprintf("invalid shape: %v", dims))
	}
	strides := make([]int, len(dims))
	stride := 1
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dims[i]
	}
	return Shape{Dims: cloneShape(dims), Strides: strides}
}

// Numel returns total number of elements. Malformed shapes report 0 so callers
// that use Numel for sizing do not accidentally pass a negative size onward.
func (s Shape) Numel() int {
	n := shapeSize(s.Dims)
	if n < 0 {
		return 0
	}
	return n
}

// Ndim returns the number of dimensions.
func (s Shape) Ndim() int { return len(s.Dims) }

// IsContiguous returns true if the tensor is stored contiguously.
func (s Shape) IsContiguous() bool {
	if len(s.Strides) != len(s.Dims) || shapeSize(s.Dims) < 0 {
		return false
	}
	stride := 1
	maxInt := int(^uint(0) >> 1)
	for i := len(s.Dims) - 1; i >= 0; i-- {
		if s.Strides[i] != stride {
			return false
		}
		if s.Dims[i] != 0 && stride > maxInt/s.Dims[i] {
			return false
		}
		stride *= s.Dims[i]
	}
	return true
}

// Reshape returns a new shape with different dimensions but same numel.
func (s Shape) Reshape(newDims []int) (Shape, error) {
	oldNumel := shapeSize(s.Dims)
	newNumel := shapeSize(newDims)
	if oldNumel < 0 || newNumel < 0 || newNumel != oldNumel {
		return Shape{}, fmt.Errorf("reshape: %v (%d) -> %v (%d)", s.Dims, oldNumel, newDims, newNumel)
	}
	return NewShape(newDims), nil
}

// Permute returns a new shape with transposed dimensions.
func (s Shape) Permute(order []int) Shape {
	if len(order) != len(s.Dims) || len(s.Strides) != len(s.Dims) {
		panic(fmt.Sprintf("permute: invalid order %v for shape %v", order, s.Dims))
	}
	seen := make([]bool, len(s.Dims))
	dims := make([]int, len(order))
	strides := make([]int, len(order))
	for i, o := range order {
		if o < 0 || o >= len(s.Dims) || seen[o] {
			panic(fmt.Sprintf("permute: invalid order %v for shape %v", order, s.Dims))
		}
		seen[o] = true
		dims[i] = s.Dims[o]
		strides[i] = s.Strides[o]
	}
	return Shape{Dims: dims, Strides: strides}
}

// Expand returns a new shape with broadcast dimensions.
func (s Shape) Expand(newDims []int) Shape {
	if len(newDims) != len(s.Dims) || len(s.Strides) != len(s.Dims) || shapeSize(newDims) < 0 {
		panic(fmt.Sprintf("expand: invalid target %v for shape %v", newDims, s.Dims))
	}
	strides := make([]int, len(newDims))
	for i := range newDims {
		if s.Dims[i] == newDims[i] {
			strides[i] = s.Strides[i]
		} else if s.Dims[i] == 1 {
			strides[i] = 0 // broadcast: stride 0
		} else {
			panic(fmt.Sprintf("expand: dim %d: %d -> %d", i, s.Dims[i], newDims[i]))
		}
	}
	return Shape{Dims: cloneShape(newDims), Strides: strides}
}

func (s Shape) String() string {
	return fmt.Sprintf("Shape%v", s.Dims)
}
