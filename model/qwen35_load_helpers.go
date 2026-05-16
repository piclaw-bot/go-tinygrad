package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/tensor"
)

func loadQwen35DenseOrNVFP4(src Qwen35TensorSource, name string, dense **tensor.Tensor, q **Qwen35NVFP4Weight, want []int) error {
	if dense == nil {
		return fmt.Errorf("nil dense destination for %s", name)
	}
	t, err := src.Get(name, want)
	if err == nil {
		*dense = t
		return nil
	}
	if q == nil {
		return err
	}
	raw, ok := unwrapQwen35RawTensorSource(src)
	if !ok {
		return err
	}
	if len(want) != 2 {
		return err
	}
	qw, qerr := LoadQwen35NVFP4WeightCandidates(raw, name, []int{want[1], want[0]})
	if qerr != nil {
		qw, qerr = LoadQwen35NVFP4WeightCandidates(raw, name, want)
	}
	if qerr != nil {
		return fmt.Errorf("%v; NVFP4 fallback: %w", err, qerr)
	}
	*q = qw
	return nil
}

func unwrapQwen35RawTensorSource(src Qwen35TensorSource) (Qwen35RawTensorSource, bool) {
	if raw, ok := src.(Qwen35RawTensorSource); ok {
		return raw, true
	}
	if cand, ok := src.(CandidateQwen35TensorSource); ok {
		if raw, ok := cand.Source.(Qwen35RawTensorSource); ok {
			return raw, true
		}
	}
	return nil, false
}
