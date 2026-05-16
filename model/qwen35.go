package model

import (
	"fmt"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

type Qwen35BaseLayerKind string

const (
	Qwen35FullAttentionLayerKind   Qwen35BaseLayerKind = "full_attention"
	Qwen35LinearAttentionLayerKind Qwen35BaseLayerKind = "linear_attention"
)

type Qwen35FullAttentionLayer struct {
	InputNorm *tensor.Tensor
	PostNorm  *tensor.Tensor
	QW        *tensor.Tensor
	KW        *tensor.Tensor
	VW        *tensor.Tensor
	OW        *tensor.Tensor
	QNorm     *tensor.Tensor
	KNorm     *tensor.Tensor
	GateW     *tensor.Tensor
	UpW       *tensor.Tensor
	DownW     *tensor.Tensor
}

type Qwen35LinearAttentionLayer struct {
	InputNorm *tensor.Tensor
	PostNorm  *tensor.Tensor
	QKVW      *tensor.Tensor
	GateW     *tensor.Tensor
	Conv1D    *tensor.Tensor
	DTBias    *tensor.Tensor
	A         *tensor.Tensor
	BetaW     *tensor.Tensor
	AlphaW    *tensor.Tensor
	Norm      *tensor.Tensor
	OutW      *tensor.Tensor
	MLPGateW  *tensor.Tensor
	MLPUpW    *tensor.Tensor
	MLPDownW  *tensor.Tensor
}

type Qwen35BaseLayer struct {
	Kind   Qwen35BaseLayerKind
	Full   *Qwen35FullAttentionLayer
	Linear *Qwen35LinearAttentionLayer
}

func ValidateQwen35FullAttentionLayer(l *Qwen35FullAttentionLayer, meta loaderconfig.QwenNativeMTPMetadata, prefix string) error {
	if l == nil {
		return fmt.Errorf("missing %s", prefix)
	}
	h := meta.HiddenSize
	inter := meta.IntermediateSize
	shapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	if err != nil {
		return err
	}
	checks := []struct {
		name string
		t    *tensor.Tensor
		want []int
	}{
		{prefix + ".input_layernorm.weight", l.InputNorm, []int{h}},
		{prefix + ".post_attention_layernorm.weight", l.PostNorm, []int{h}},
		{prefix + ".self_attn.q_proj.weight", l.QW, shapes.QProj},
		{prefix + ".self_attn.k_proj.weight", l.KW, shapes.KProj},
		{prefix + ".self_attn.v_proj.weight", l.VW, shapes.VProj},
		{prefix + ".self_attn.o_proj.weight", l.OW, shapes.OProj},
		{prefix + ".self_attn.q_norm.weight", l.QNorm, shapes.QNorm},
		{prefix + ".self_attn.k_norm.weight", l.KNorm, shapes.KNorm},
		{prefix + ".mlp.gate_proj.weight", l.GateW, []int{inter, h}},
		{prefix + ".mlp.up_proj.weight", l.UpW, []int{inter, h}},
		{prefix + ".mlp.down_proj.weight", l.DownW, []int{h, inter}},
	}
	for _, c := range checks {
		if err := expectShape(c.t, c.want, c.name); err != nil {
			return err
		}
	}
	return nil
}

func ValidateQwen35LinearAttentionLayer(l *Qwen35LinearAttentionLayer, meta loaderconfig.QwenNativeMTPMetadata, prefix string) error {
	if l == nil {
		return fmt.Errorf("missing %s", prefix)
	}
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		return err
	}
	h := meta.HiddenSize
	inter := meta.IntermediateSize
	checks := []struct {
		name string
		t    *tensor.Tensor
		want []int
	}{
		{prefix + ".input_layernorm.weight", l.InputNorm, []int{h}},
		{prefix + ".post_attention_layernorm.weight", l.PostNorm, []int{h}},
		{prefix + ".linear_attn.in_proj_qkvz.weight", l.QKVW, shapes.QKV},
		{prefix + ".linear_attn.in_proj_gate.weight", l.GateW, shapes.Gate},
		{prefix + ".linear_attn.conv1d.weight", l.Conv1D, shapes.Conv1D},
		{prefix + ".linear_attn.dt_bias", l.DTBias, shapes.DTBias},
		{prefix + ".linear_attn.A", l.A, shapes.A},
		{prefix + ".linear_attn.in_proj_ba.weight", l.BetaW, shapes.Beta},
		{prefix + ".linear_attn.in_proj_a.weight", l.AlphaW, shapes.Alpha},
		{prefix + ".linear_attn.norm.weight", l.Norm, shapes.Norm},
		{prefix + ".linear_attn.out_proj.weight", l.OutW, shapes.Out},
		{prefix + ".mlp.gate_proj.weight", l.MLPGateW, []int{inter, h}},
		{prefix + ".mlp.up_proj.weight", l.MLPUpW, []int{inter, h}},
		{prefix + ".mlp.down_proj.weight", l.MLPDownW, []int{h, inter}},
	}
	for _, c := range checks {
		if err := expectShape(c.t, c.want, c.name); err != nil {
			return err
		}
	}
	return nil
}

func qwen35LinearAttentionShapesFromMeta(meta loaderconfig.QwenNativeMTPMetadata) (loaderconfig.Qwen35LinearAttentionShapes, error) {
	ssmInner := meta.LinearNumValueHeads * meta.LinearValueHeadDim
	ssmState := meta.LinearKeyHeadDim
	return loaderconfig.Qwen35LinearAttentionShapesFor(meta.HiddenSize, ssmInner, ssmState, meta.LinearConvKernelDim, meta.LinearNumValueHeads, meta.LinearNumKeyHeads)
}
