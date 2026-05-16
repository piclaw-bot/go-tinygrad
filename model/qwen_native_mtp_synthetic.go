package model

import (
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

// NewSyntheticQwenNativeMTPFixture returns a tiny deterministic fixture for
// command-line correctness harnesses. It is not a model-quality fixture; it only
// exercises native-MTP plumbing without loading Qwen3.6 weights.
func NewSyntheticQwenNativeMTPFixture() (*LlamaModel, *QwenNativeMTPHead, loaderconfig.QwenNativeMTPMetadata, QwenNativeMTPDraftState) {
	meta := loaderconfig.QwenNativeMTPMetadata{
		HiddenSize:         4,
		IntermediateSize:   6,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		MTPNumHiddenLayers: 1,
		HasNativeMTP:       true,
	}
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 2, HiddenSize: meta.HiddenSize, RMSNormEps: 1e-6},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		}, []int{2, 4}),
		LMHead: tensor.FromFloat32([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		}, []int{2, 4}),
	}
	head := syntheticQwenNativeMTPHeadForFixture(meta)
	state := QwenNativeMTPDraftState{Hidden: []float32{0, 1, 0, 0}}
	return m, head, meta, state
}

func syntheticQwenNativeMTPHeadForFixture(meta loaderconfig.QwenNativeMTPMetadata) *QwenNativeMTPHead {
	h := meta.HiddenSize
	inter := meta.IntermediateSize
	attn, _ := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	return &QwenNativeMTPHead{
		FC:                 tensor.Zeros([]int{h, 2 * h}),
		PreFCNormEmbedding: tensor.Ones([]int{h}),
		PreFCNormHidden:    tensor.Ones([]int{h}),
		Norm:               tensor.Ones([]int{h}),
		Layers: []QwenNativeMTPLayer{{
			InputNorm: tensor.Ones([]int{h}),
			PostNorm:  tensor.Ones([]int{h}),
			QW:        tensor.Zeros(attn.QProj),
			KW:        tensor.Zeros(attn.KProj),
			VW:        tensor.Zeros(attn.VProj),
			OW:        tensor.Zeros(attn.OProj),
			QNorm:     tensor.Ones(attn.QNorm),
			KNorm:     tensor.Ones(attn.KNorm),
			GateW:     tensor.Zeros([]int{inter, h}),
			UpW:       tensor.Zeros([]int{inter, h}),
			DownW:     tensor.Zeros([]int{h, inter}),
		}},
	}
}
