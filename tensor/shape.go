package tensor

import "fmt"

// Shape represents a tensor's dimensions with strides for views.
type Shape struct {
	Dims    []int
	Strides []int
}

// NewShape creates a contiguous shape.
func NewShape(dims []int) Shape {
	strides := make([]int, len(dims))
	stride := 1
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dims[i]
	}
	return Shape{Dims: cloneShape(dims), Strides: strides}
}

// Numel returns total number of elements.
func (s Shape) Numel() int { return shapeSize(s.Dims) }

// Ndim returns the number of dimensions.
func (s Shape) Ndim() int { return len(s.Dims) }

// IsContiguous returns true if the tensor is stored contiguously.
func (s Shape) IsContiguous() bool {
	stride := 1
	for i := len(s.Dims) - 1; i >= 0; i-- {
		if s.Strides[i] != stride {
			return false
		}
		stride *= s.Dims[i]
	}
	return true
}

// Reshape returns a new shape with different dimensions but same numel.
func (s Shape) Reshape(newDims []int) (Shape, error) {
	if shapeSize(newDims) != s.Numel() {
		return Shape{}, fmt.Errorf("reshape: %v (%d) -> %v (%d)", s.Dims, s.Numel(), newDims, shapeSize(newDims))
	}
	return NewShape(newDims), nil
}

// Permute returns a new shape with transposed dimensions.
func (s Shape) Permute(order []int) Shape {
	dims := make([]int, len(order))
	strides := make([]int, len(order))
	for i, o := range order {
		dims[i] = s.Dims[o]
		strides[i] = s.Strides[o]
	}
	return Shape{Dims: dims, Strides: strides}
}

// Expand returns a new shape with broadcast dimensions.
func (s Shape) Expand(newDims []int) Shape {
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
