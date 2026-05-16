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

func TestAppendQwen35FullAttentionKV(t *testing.T) {
	meta := testQwen35BaseMeta()
	nextK, nextV, err := appendQwen35FullAttentionKV([]float32{1, 2}, []float32{3, 4}, []float32{5, 6}, []float32{7, 8}, meta)
	if err != nil {
		t.Fatalf("appendQwen35FullAttentionKV: %v", err)
	}
	if len(nextK) != 4 || len(nextV) != 4 || nextK[2] != 5 || nextV[3] != 8 {
		t.Fatalf("next K/V=%v/%v", nextK, nextV)
	}
	if _, _, err := appendQwen35FullAttentionKV([]float32{1}, []float32{1}, []float32{1, 2}, []float32{1, 2}, meta); err == nil {
		t.Fatal("bad past KV multiple returned nil error")
	}
	if _, _, err := appendQwen35FullAttentionKV(nil, nil, []float32{1}, []float32{1}, meta); err == nil {
		t.Fatal("bad current KV len returned nil error")
	}
}

func TestQwen35BaseModelForwardOneFullAttention(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"full_attention"}
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.layers.0")}
	base, err := LoadQwen35BaseModelLayers(src, meta)
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35BaseForwardState(base, meta)
	if err != nil {
		t.Fatal(err)
	}
	out, next, err := base.ForwardOne([]float32{1, 0, 0, 0}, state, 0, nil, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardOne: %v", err)
	}
	if len(out) != meta.HiddenSize || next.Pos != 1 || len(next.FullK[0]) != meta.NumKeyValueHeads*meta.HeadDim {
		t.Fatalf("out=%v next=%+v", out, next)
	}
}

func TestQwen35BaseModelForwardOneLinearUnsupported(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"linear_attention"}
	src := CandidateQwen35TensorSource{Source: linearQwen35LayerSource(meta, "model.layers.0")}
	base, err := LoadQwen35BaseModelLayers(src, meta)
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35BaseForwardState(base, meta)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = base.ForwardOne([]float32{1, 0, 0, 0}, state, 0, nil, 1e-6, meta)
	if err == nil || !strings.Contains(err.Error(), "linear-attention layer 0") || !strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("expected linear unsupported, got %v", err)
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

func TestSplitQwen35LinearQKVZ(t *testing.T) {
	meta := testQwen35BaseMeta()
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		t.Fatal(err)
	}
	projected := make([]float32, shapes.ValueDim+shapes.KeyDim+shapes.ValueDim+shapes.ValueDim)
	for i := range projected {
		projected[i] = float32(i + 1)
	}
	parts, err := splitQwen35LinearQKVZ(projected, shapes)
	if err != nil {
		t.Fatalf("splitQwen35LinearQKVZ: %v", err)
	}
	if len(parts.Q) != shapes.ValueDim || len(parts.K) != shapes.KeyDim || len(parts.V) != shapes.ValueDim || len(parts.Z) != shapes.ValueDim {
		t.Fatalf("parts lens Q/K/V/Z=%d/%d/%d/%d", len(parts.Q), len(parts.K), len(parts.V), len(parts.Z))
	}
	if parts.Q[0] != 1 || parts.K[0] != float32(shapes.ValueDim+1) {
		t.Fatalf("unexpected split parts=%+v", parts)
	}
	if _, err := splitQwen35LinearQKVZ(projected[:len(projected)-1], shapes); err == nil {
		t.Fatal("bad projected length returned nil error")
	}
}

func TestApplyQwen35LinearDepthwiseConv(t *testing.T) {
	out, err := applyQwen35LinearDepthwiseConv(
		[]float32{1, 2, 3, 4, 5, 6},
		[]float32{1, 10, 1, 10, 1, 10},
		2,
		3,
	)
	if err != nil {
		t.Fatalf("applyQwen35LinearDepthwiseConv: %v", err)
	}
	if len(out) != 2 || out[0] != 9 || out[1] != 120 {
		t.Fatalf("out=%v", out)
	}
	if _, err := applyQwen35LinearDepthwiseConv([]float32{1}, []float32{1}, 0, 1); err == nil {
		t.Fatal("bad dims returned nil error")
	}
	if _, err := applyQwen35LinearDepthwiseConv([]float32{1, 2}, []float32{1}, 1, 2); err == nil {
		t.Fatal("bad weight len returned nil error")
	}
}

func TestUpdateQwen35LinearConvState(t *testing.T) {
	next, err := updateQwen35LinearConvState([]float32{1, 2, 3, 4, 5, 6}, []float32{7, 8}, 3)
	if err != nil {
		t.Fatalf("updateQwen35LinearConvState: %v", err)
	}
	want := []float32{3, 4, 5, 6, 7, 8}
	for i := range want {
		if next[i] != want[i] {
			t.Fatalf("next=%v want %v", next, want)
		}
	}
	if _, err := updateQwen35LinearConvState([]float32{1, 2}, []float32{3, 4}, 0); err == nil {
		t.Fatal("bad kernel returned nil error")
	}
	if _, err := updateQwen35LinearConvState([]float32{1, 2}, []float32{3, 4}, 2); err == nil {
		t.Fatal("bad state len returned nil error")
	}
}

func TestNewQwen35LinearAttentionState(t *testing.T) {
	meta := testQwen35BaseMeta()
	state, err := NewQwen35LinearAttentionState(meta)
	if err != nil {
		t.Fatalf("NewQwen35LinearAttentionState: %v", err)
	}
	shapes, _ := qwen35LinearAttentionShapesFromMeta(meta)
	if len(state.Conv) != shapes.ConvDim*meta.LinearConvKernelDim {
		t.Fatalf("conv len=%d", len(state.Conv))
	}
	wantSSM := meta.LinearNumValueHeads * meta.LinearValueHeadDim * meta.LinearNumKeyHeads * meta.LinearKeyHeadDim
	if len(state.SSM) != wantSSM {
		t.Fatalf("ssm len=%d want %d", len(state.SSM), wantSSM)
	}
}

func TestQwen35LinearAttentionForwardStub(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := CandidateQwen35TensorSource{Source: linearQwen35LayerSource(meta, "model.layers.1")}
	l, err := LoadQwen35LinearAttentionLayer(src, meta, "model.layers.1")
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35LinearAttentionState(meta)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = l.ForwardWithState([]float32{1, 0, 0, 0}, state, 1e-6, meta)
	if err == nil || !strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("expected not implemented, got %v", err)
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
