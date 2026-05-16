package model

import (
	"fmt"
	"strings"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

type Qwen35TensorSource interface {
	Get(name string, shape []int) (*tensor.Tensor, error)
}

type CandidateQwen35TensorSource struct {
	Source Qwen35TensorSource
}

func (s CandidateQwen35TensorSource) Get(name string, shape []int) (*tensor.Tensor, error) {
	if s.Source == nil {
		return nil, fmt.Errorf("nil Qwen3.5 tensor source for %s", name)
	}
	var errs []string
	for _, candidate := range loaderconfig.Qwen35TensorNameCandidates(name) {
		t, err := s.Source.Get(candidate, shape)
		if err == nil {
			return t, nil
		}
		errs = append(errs, fmt.Sprintf("%s: %v", candidate, err))
	}
	return nil, fmt.Errorf("load %s: no Qwen3.5 tensor-name candidate matched (%s)", name, strings.Join(errs, "; "))
}
