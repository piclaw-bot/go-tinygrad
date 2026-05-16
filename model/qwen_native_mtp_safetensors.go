package model

import (
	"fmt"
	"path/filepath"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/tensor"
)

type SafetensorsQwenNativeMTPTensorSource struct {
	File *safetensors.File
}

type ShardedSafetensorsQwenNativeMTPTensorSource struct {
	File *safetensors.ShardedFile
}

type qwenNativeMTPClosableTensorSource interface {
	QwenNativeMTPTensorSource
	Close() error
}

type qwenNativeMTPSingleFileSource struct {
	SafetensorsQwenNativeMTPTensorSource
}

type qwenNativeMTPShardedFileSource struct {
	ShardedSafetensorsQwenNativeMTPTensorSource
}

func OpenQwenNativeMTPSafetensorsSource(dir string) (qwenNativeMTPClosableTensorSource, error) {
	if sf, err := safetensors.OpenSharded(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		return qwenNativeMTPShardedFileSource{ShardedSafetensorsQwenNativeMTPTensorSource{File: sf}}, nil
	}
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	return qwenNativeMTPSingleFileSource{SafetensorsQwenNativeMTPTensorSource{File: f}}, nil
}

func LoadQwenNativeMTPHeadFromSafetensorsDir(dir string, meta loaderconfig.QwenNativeMTPMetadata) (*QwenNativeMTPHead, error) {
	src, err := OpenQwenNativeMTPSafetensorsSource(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadQwenNativeMTPHead(src, meta)
}

func LoadQwen35BaseModelLayersFromSafetensorsDir(dir string, meta loaderconfig.QwenNativeMTPMetadata) (*Qwen35BaseModel, error) {
	src, err := OpenQwenNativeMTPSafetensorsSource(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadQwen35BaseModelLayers(CandidateQwen35TensorSource{Source: src}, meta)
}

func (s qwenNativeMTPSingleFileSource) Close() error {
	if s.File == nil {
		return nil
	}
	return s.File.Close()
}

func (s qwenNativeMTPShardedFileSource) Close() error {
	if s.File == nil {
		return nil
	}
	return s.File.Close()
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

func (s ShardedSafetensorsQwenNativeMTPTensorSource) Get(name string, shape []int) (*tensor.Tensor, error) {
	if s.File == nil {
		return nil, fmt.Errorf("nil sharded safetensors file for %s", name)
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
