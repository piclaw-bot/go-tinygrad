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

func fullQwen35LayerSource(meta loaderconfig.QwenNativeMTPMetadata, prefix string) fakeQwen35TensorSource {
	shapes, _ := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	return fakeQwen35TensorSource{
		prefix + ".input_layernorm.weight":          tensor.Ones([]int{4}),
		prefix + ".post_attention_layernorm.weight": tensor.Ones([]int{4}),
		prefix + ".self_attn.q_proj.weight":         tensor.Zeros(shapes.QProj),
		prefix + ".self_attn.k_proj.weight":         tensor.Zeros(shapes.KProj),
		prefix + ".self_attn.v_proj.weight":         tensor.Zeros(shapes.VProj),
		prefix + ".self_attn.o_proj.weight":         tensor.Zeros(shapes.OProj),
		prefix + ".self_attn.q_norm.weight":         tensor.Ones(shapes.QNorm),
		prefix + ".self_attn.k_norm.weight":         tensor.Ones(shapes.KNorm),
		prefix + ".mlp.gate_proj.weight":            tensor.Zeros([]int{6, 4}),
		prefix + ".mlp.up_proj.weight":              tensor.Zeros([]int{6, 4}),
		prefix + ".mlp.down_proj.weight":            tensor.Zeros([]int{4, 6}),
	}
}

func TestLoadQwen35BaseModelLayers(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 2
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"linear_attention", "full_attention"}
	src := fakeQwen35TensorSource{}
	for k, v := range linearQwen35LayerSource(meta, "model.language_model.model.layers.0") {
		src[k] = v
	}
	for k, v := range fullQwen35LayerSource(meta, "model.language_model.model.layers.1") {
		src[k] = v
	}
	model, err := LoadQwen35BaseModelLayers(CandidateQwen35TensorSource{Source: src}, meta)
	if err != nil {
		t.Fatalf("LoadQwen35BaseModelLayers: %v", err)
	}
	if len(model.Layers) != 2 || model.Layers[0].Kind != Qwen35LinearAttentionLayerKind || model.Layers[1].Kind != Qwen35FullAttentionLayerKind {
		t.Fatalf("layers=%+v", model.Layers)
	}
}

func TestLoadQwen35FullAttentionLayer(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.language_model.model.layers.0")}
	l, err := LoadQwen35FullAttentionLayer(src, meta, "model.layers.0")
	if err != nil {
		t.Fatalf("LoadQwen35FullAttentionLayer: %v", err)
	}
	if l.QW == nil || l.GateW == nil {
		t.Fatalf("loaded layer=%+v", l)
	}
}

func TestQwen35FullAttentionLayerForward(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.layers.0")}
	l, err := LoadQwen35FullAttentionLayer(src, meta, "model.layers.0")
	if err != nil {
		t.Fatal(err)
	}
	out, curK, curV, err := l.ForwardWithKV([]float32{1, 0, 0, 0}, 0, nil, nil, nil, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardWithKV: %v", err)
	}
	if len(out) != meta.HiddenSize || len(curK) != meta.NumKeyValueHeads*meta.HeadDim || len(curV) != meta.NumKeyValueHeads*meta.HeadDim {
		t.Fatalf("out/K/V lens=%d/%d/%d", len(out), len(curK), len(curV))
	}
	if out[0] == 0 {
		t.Fatalf("expected residual output, got %v", out)
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

func linearQwen35LayerSource(meta loaderconfig.QwenNativeMTPMetadata, prefix string) fakeQwen35TensorSource {
	shapes, _ := qwen35LinearAttentionShapesFromMeta(meta)
	return fakeQwen35TensorSource{
		prefix + ".input_layernorm.weight":          tensor.Ones([]int{4}),
		prefix + ".post_attention_layernorm.weight": tensor.Ones([]int{4}),
		prefix + ".linear_attn.in_proj_qkvz.weight": tensor.Zeros(shapes.QKV),
		prefix + ".linear_attn.in_proj_gate.weight": tensor.Zeros(shapes.Gate),
		prefix + ".linear_attn.conv1d.weight":       tensor.Zeros(shapes.Conv1D),
		prefix + ".linear_attn.dt_bias":             tensor.Zeros(shapes.DTBias),
		prefix + ".linear_attn.A":                   tensor.Zeros(shapes.A),
		prefix + ".linear_attn.in_proj_ba.weight":   tensor.Zeros(shapes.Beta),
		prefix + ".linear_attn.in_proj_a.weight":    tensor.Zeros(shapes.Alpha),
		prefix + ".linear_attn.norm.weight":         tensor.Ones(shapes.Norm),
		prefix + ".linear_attn.out_proj.weight":     tensor.Zeros(shapes.Out),
		prefix + ".mlp.gate_proj.weight":            tensor.Zeros([]int{6, 4}),
		prefix + ".mlp.up_proj.weight":              tensor.Zeros([]int{6, 4}),
		prefix + ".mlp.down_proj.weight":            tensor.Zeros([]int{4, 6}),
	}
}

func TestLoadQwen35LinearAttentionLayer(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := CandidateQwen35TensorSource{Source: linearQwen35LayerSource(meta, "model.language_model.model.layers.1")}
	l, err := LoadQwen35LinearAttentionLayer(src, meta, "model.layers.1")
	if err != nil {
		t.Fatalf("LoadQwen35LinearAttentionLayer: %v", err)
	}
	if l.QKVW == nil || l.OutW == nil {
		t.Fatalf("loaded layer=%+v", l)
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
