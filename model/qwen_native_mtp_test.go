package model

import (
	"strings"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestValidateQwenNativeMTPHeadSynthetic(t *testing.T) {
	meta := loaderconfig.QwenNativeMTPMetadata{
		HiddenSize:         4,
		IntermediateSize:   6,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		MTPNumHiddenLayers: 1,
		HasNativeMTP:       true,
	}
	head := syntheticQwenNativeMTPHead(meta)
	if err := ValidateQwenNativeMTPHead(head, meta); err != nil {
		t.Fatalf("ValidateQwenNativeMTPHead: %v", err)
	}
	head.Layers[0].QW = tensor.Zeros([]int{1, 1})
	if err := ValidateQwenNativeMTPHead(head, meta); err == nil || !strings.Contains(err.Error(), "q_proj") {
		t.Fatalf("bad q_proj validation err=%v", err)
	}
}

func syntheticQwenNativeMTPHead(meta loaderconfig.QwenNativeMTPMetadata) *QwenNativeMTPHead {
	h := meta.HiddenSize
	inter := meta.IntermediateSize
	attn, _ := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	return &QwenNativeMTPHead{
		FC:                 tensor.Zeros([]int{h, 2 * h}),
		PreFCNormEmbedding: tensor.Zeros([]int{h}),
		PreFCNormHidden:    tensor.Zeros([]int{h}),
		Norm:               tensor.Zeros([]int{h}),
		Layers: []QwenNativeMTPLayer{{
			InputNorm: tensor.Zeros([]int{h}),
			PostNorm:  tensor.Zeros([]int{h}),
			QW:        tensor.Zeros(attn.QProj),
			KW:        tensor.Zeros(attn.KProj),
			VW:        tensor.Zeros(attn.VProj),
			OW:        tensor.Zeros(attn.OProj),
			QNorm:     tensor.Zeros(attn.QNorm),
			KNorm:     tensor.Zeros(attn.KNorm),
			GateW:     tensor.Zeros([]int{inter, h}),
			UpW:       tensor.Zeros([]int{inter, h}),
			DownW:     tensor.Zeros([]int{h, inter}),
		}},
	}
}
