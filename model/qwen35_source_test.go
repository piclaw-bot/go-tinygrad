package model

import (
	"fmt"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

type fakeQwen35TensorSource map[string]*tensor.Tensor

func (f fakeQwen35TensorSource) Get(name string, shape []int) (*tensor.Tensor, error) {
	t, ok := f[name]
	if !ok {
		return nil, fmt.Errorf("missing %s", name)
	}
	if err := expectShape(t, shape, name); err != nil {
		return nil, err
	}
	return t, nil
}

func TestCandidateQwen35TensorSourceNestedName(t *testing.T) {
	src := CandidateQwen35TensorSource{Source: fakeQwen35TensorSource{
		"model.language_model.model.layers.0.self_attn.q_proj.weight": tensor.Zeros([]int{8, 4}),
	}}
	got, err := src.Get("model.layers.0.self_attn.q_proj.weight", []int{8, 4})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatal("nil tensor")
	}
}

func TestCandidateQwen35TensorSourceMTPNameNotNested(t *testing.T) {
	src := CandidateQwen35TensorSource{Source: fakeQwen35TensorSource{
		"model.language_model.mtp.fc.weight": tensor.Zeros([]int{4, 8}),
	}}
	_, err := src.Get("model.language_model.mtp.fc.weight", []int{4, 8})
	if err == nil || !strings.Contains(err.Error(), "mtp.fc.weight") {
		t.Fatalf("expected normalized MTP-only miss, got %v", err)
	}
}

func TestCandidateQwen35TensorSourceMissingListsCandidates(t *testing.T) {
	src := CandidateQwen35TensorSource{Source: fakeQwen35TensorSource{}}
	_, err := src.Get("model.layers.0.self_attn.q_proj.weight", []int{8, 4})
	if err == nil || !strings.Contains(err.Error(), "model.language_model.model.layers.0.self_attn.q_proj.weight") {
		t.Fatalf("candidate error=%v", err)
	}
}
