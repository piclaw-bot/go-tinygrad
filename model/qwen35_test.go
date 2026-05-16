package model

import (
	"strings"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

func testQwen35BaseMeta() loaderconfig.QwenNativeMTPMetadata {
	return loaderconfig.QwenNativeMTPMetadata{
		HiddenSize:            4,
		IntermediateSize:      6,
		NumAttentionHeads:     2,
		NumKeyValueHeads:      1,
		HeadDim:               2,
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      2,
		LinearNumKeyHeads:     1,
		LinearNumValueHeads:   2,
		LinearValueHeadDim:    2,
		HasLinearAttention:    true,
		FullAttentionInterval: 4,
	}
}

func TestValidateQwen35FullAttentionLayer(t *testing.T) {
	meta := testQwen35BaseMeta()
	shapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	l := &Qwen35FullAttentionLayer{
		InputNorm: tensor.Ones([]int{4}), PostNorm: tensor.Ones([]int{4}),
		QW: tensor.Zeros(shapes.QProj), KW: tensor.Zeros(shapes.KProj), VW: tensor.Zeros(shapes.VProj), OW: tensor.Zeros(shapes.OProj),
		QNorm: tensor.Ones(shapes.QNorm), KNorm: tensor.Ones(shapes.KNorm),
		GateW: tensor.Zeros([]int{6, 4}), UpW: tensor.Zeros([]int{6, 4}), DownW: tensor.Zeros([]int{4, 6}),
	}
	if err := ValidateQwen35FullAttentionLayer(l, meta, "model.layers.0"); err != nil {
		t.Fatalf("ValidateQwen35FullAttentionLayer: %v", err)
	}
	l.QW = tensor.Zeros([]int{1, 4})
	if err := ValidateQwen35FullAttentionLayer(l, meta, "model.layers.0"); err == nil || !strings.Contains(err.Error(), "q_proj") {
		t.Fatalf("bad q_proj error=%v", err)
	}
}

func TestValidateQwen35LinearAttentionLayer(t *testing.T) {
	meta := testQwen35BaseMeta()
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		t.Fatal(err)
	}
	l := &Qwen35LinearAttentionLayer{
		InputNorm: tensor.Ones([]int{4}), PostNorm: tensor.Ones([]int{4}),
		QKVW: tensor.Zeros(shapes.QKV), GateW: tensor.Zeros(shapes.Gate), Conv1D: tensor.Zeros(shapes.Conv1D),
		DTBias: tensor.Zeros(shapes.DTBias), A: tensor.Zeros(shapes.A), BetaW: tensor.Zeros(shapes.Beta), AlphaW: tensor.Zeros(shapes.Alpha),
		Norm: tensor.Ones(shapes.Norm), OutW: tensor.Zeros(shapes.Out),
		MLPGateW: tensor.Zeros([]int{6, 4}), MLPUpW: tensor.Zeros([]int{6, 4}), MLPDownW: tensor.Zeros([]int{4, 6}),
	}
	if err := ValidateQwen35LinearAttentionLayer(l, meta, "model.layers.1"); err != nil {
		t.Fatalf("ValidateQwen35LinearAttentionLayer: %v", err)
	}
	l.Conv1D = tensor.Zeros([]int{1, shapes.ConvDim})
	if err := ValidateQwen35LinearAttentionLayer(l, meta, "model.layers.1"); err == nil || !strings.Contains(err.Error(), "conv1d") {
		t.Fatalf("bad conv1d error=%v", err)
	}
}
