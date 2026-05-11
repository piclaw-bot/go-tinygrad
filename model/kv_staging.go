package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

// LayerKVDim returns the per-token K/V vector width appended by one layer.
// Shared-KV layers return 0 because they reuse a source layer and do not append.
func (m *LlamaModel) LayerKVDim(layerIdx int) (int, error) {
	if m == nil {
		return 0, fmt.Errorf("nil model")
	}
	if layerIdx < 0 || layerIdx >= len(m.Layers) {
		return 0, fmt.Errorf("layer index %d out of range [0,%d)", layerIdx, len(m.Layers))
	}
	layer := m.Layers[layerIdx]
	if !layer.HasKV {
		return 0, nil
	}
	if m.Config.NumKVHeads <= 0 {
		return 0, fmt.Errorf("num_key_value_heads=%d", m.Config.NumKVHeads)
	}
	headDim := m.Config.HeadDim
	if layer.HeadDimLocal > 0 {
		headDim = layer.HeadDimLocal
	}
	if headDim <= 0 {
		return 0, fmt.Errorf("layer %d head_dim=%d", layerIdx, headDim)
	}
	return m.Config.NumKVHeads * headDim, nil
}

// LayerKVDims returns per-layer K/V widths suitable for FloatKVCheckpoint
// keep-prefix commits. Layers that do not append K/V have dimension 0.
func (m *LlamaModel) LayerKVDims() ([]int, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	dims := make([]int, len(m.Layers))
	for i := range m.Layers {
		dim, err := m.LayerKVDim(i)
		if err != nil {
			return nil, err
		}
		dims[i] = dim
	}
	return dims, nil
}

// CommitAcceptedFloatKV keeps the accepted verifier KV prefix plus bonus token
// using this model's per-layer K/V widths.
func (m *LlamaModel) CommitAcceptedFloatKV(kvCacheK, kvCacheV [][]float32, cp kv.FloatKVCheckpoint, acceptance MTPAcceptance) error {
	dims, err := m.LayerKVDims()
	if err != nil {
		return err
	}
	return CommitAcceptedFloatKV(kvCacheK, kvCacheV, cp, dims, acceptance)
}
