package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestForwardLayerRejectsMalformedInputs(t *testing.T) {
	if got := (*LlamaModel)(nil).ForwardLayer([]float32{1}, 0, 0, 0, nil, nil); got != nil {
		t.Fatalf("nil model output=%v, want nil", got)
	}
	m := &LlamaModel{Config: LlamaConfig{HiddenSize: 2, NumHeads: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{}}}
	if got := m.ForwardLayer([]float32{1, 2}, -1, 0, 0, nil, nil); got != nil {
		t.Fatalf("bad layer index output=%v, want nil", got)
	}
	if got := m.ForwardLayer([]float32{1}, 0, 0, 0, nil, nil); got != nil {
		t.Fatalf("short hidden output=%v, want nil", got)
	}
	if got := m.ForwardLayer([]float32{1, 2}, 0, 0, -1, nil, nil); got != nil {
		t.Fatalf("negative pos output=%v, want nil", got)
	}
	m.Layers[0].InputNorm = tensor.Ones([]int{2})
	m.Layers[0].PostNorm = tensor.Ones([]int{2})
	if got := m.ForwardLayer([]float32{1, 2}, 0, 0, 0, nil, nil); got != nil {
		t.Fatalf("missing KV cache output=%v, want nil", got)
	}
	m.Config.NumHeads = int(^uint(0) >> 1)
	if got := m.ForwardLayer([]float32{1, 2}, 0, 0, 0, make([][]float32, 1), make([][]float32, 1)); got != nil {
		t.Fatalf("overflowing qDim output=%v, want nil", got)
	}
	m.Config.NumHeads = 1
	m.Layers[0].HasKV = true
	m.Layers[0].QNorm = tensor.Ones([]int{2})
	m.Layers[0].KNorm = nil
	if got := m.ForwardLayer([]float32{1, 2}, 0, 0, 0, make([][]float32, 1), make([][]float32, 1)); got != nil {
		t.Fatalf("missing KNorm output=%v, want nil", got)
	}
}
