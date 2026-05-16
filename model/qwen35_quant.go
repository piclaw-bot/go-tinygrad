package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/runtime/quant"
)

type Qwen35RawTensorSource interface {
	GetRaw(name string) ([]byte, string, []int, error)
}

type Qwen35NVFP4Weight struct {
	Name string
	W    *quant.NVFP4Weight
}

func LoadQwen35NVFP4WeightCandidates(src Qwen35RawTensorSource, name string, wantShape []int) (*Qwen35NVFP4Weight, error) {
	var lastErr error
	for _, candidate := range loaderconfig.Qwen35TensorNameCandidates(name) {
		w, err := LoadQwen35NVFP4Weight(src, candidate, wantShape)
		if err == nil {
			return w, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("load %s NVFP4: %w", name, lastErr)
}

func LoadQwen35NVFP4Weight(src Qwen35RawTensorSource, name string, wantShape []int) (*Qwen35NVFP4Weight, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Qwen3.5 raw tensor source")
	}
	if len(wantShape) != 2 || wantShape[0] <= 0 || wantShape[1] <= 0 {
		return nil, fmt.Errorf("invalid Qwen3.5 NVFP4 wanted shape %v", wantShape)
	}
	actualName := name
	raw, dtype, shape, err := src.GetRaw(actualName)
	if err != nil {
		return nil, err
	}
	if dtype != "U8" {
		return nil, fmt.Errorf("%s dtype=%s, want U8 NVFP4", actualName, dtype)
	}
	if len(shape) != 2 || shape[0] != wantShape[0] || shape[1]*2 != wantShape[1] {
		return nil, fmt.Errorf("%s packed shape=%v want [%d %d/2]", actualName, shape, wantShape[0], wantShape[1])
	}
	prefix := strings.TrimSuffix(actualName, ".weight")
	scale, scaleDType, scaleShape, err := src.GetRaw(prefix + ".weight_scale")
	if err != nil {
		return nil, err
	}
	if scaleDType != "F8_E4M3" {
		return nil, fmt.Errorf("%s.weight_scale dtype=%s, want F8_E4M3", prefix, scaleDType)
	}
	groups := wantShape[1] / 16
	if len(scaleShape) != 2 || scaleShape[0] != wantShape[0] || scaleShape[1] != groups {
		return nil, fmt.Errorf("%s.weight_scale shape=%v want [%d %d]", prefix, scaleShape, wantShape[0], groups)
	}
	scale2, err := loadQwen35ScalarF32(src, prefix+".weight_scale_2")
	if err != nil {
		return nil, err
	}
	inputScale, inputErr := loadQwen35ScalarF32(src, prefix+".input_scale")
	qw := &quant.NVFP4Weight{Weight: append([]byte(nil), raw...), WeightScale: append([]byte(nil), scale...), WeightScale2: scale2, OutDim: wantShape[0], InDim: wantShape[1], Groups: groups, GroupSize: 16}
	if inputErr == nil {
		qw.InputScale = inputScale
		qw.HasInputScale = true
	}
	if err := quant.ValidateNVFP4Weight(qw); err != nil {
		return nil, err
	}
	return &Qwen35NVFP4Weight{Name: actualName, W: qw}, nil
}

func loadQwen35ScalarF32(src Qwen35RawTensorSource, name string) (float32, error) {
	raw, dtype, shape, err := src.GetRaw(name)
	if err != nil {
		return 0, err
	}
	if dtype != "F32" {
		return 0, fmt.Errorf("%s dtype=%s, want F32", name, dtype)
	}
	if len(shape) != 0 || len(raw) != 4 {
		return 0, fmt.Errorf("%s scalar shape=%v raw_len=%d", name, shape, len(raw))
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(raw)), nil
}
