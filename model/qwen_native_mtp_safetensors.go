package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/tensor"
)

type SafetensorsQwenNativeMTPTensorSource struct {
	File *safetensors.File
}

func (s SafetensorsQwenNativeMTPTensorSource) Get(name string, shape []int) (*tensor.Tensor, error) {
	if s.File == nil {
		return nil, fmt.Errorf("nil safetensors file for %s", name)
	}
	data, actualShape, err := s.File.GetFloat32(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if len(actualShape) > 0 {
		if sameQwenMTPShape(actualShape, shape) {
			shape = actualShape
		} else {
			return nil, fmt.Errorf("%s shape=%v want %v", name, actualShape, shape)
		}
	}
	return tensor.FromFloat32(data, shape), nil
}

func sameQwenMTPShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
