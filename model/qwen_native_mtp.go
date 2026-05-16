package model

import (
	"fmt"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

// QwenNativeMTPHead describes the native in-model MTP head used by
// Qwen3.5/Qwen3.6 text checkpoints. It is intentionally separate from the
// Gemma4 assistant-drafter structures.
type QwenNativeMTPHead struct {
	FC                 *tensor.Tensor
	PreFCNormEmbedding *tensor.Tensor
	PreFCNormHidden    *tensor.Tensor
	Norm               *tensor.Tensor
	Layers             []QwenNativeMTPLayer
}

type QwenNativeMTPLayer struct {
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

type QwenNativeMTPTensorSource interface {
	Get(name string, shape []int) (*tensor.Tensor, error)
}

func LoadQwenNativeMTPHead(src QwenNativeMTPTensorSource, meta loaderconfig.QwenNativeMTPMetadata) (*QwenNativeMTPHead, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Qwen native MTP tensor source")
	}
	h := meta.HiddenSize
	inter := meta.IntermediateSize
	head := &QwenNativeMTPHead{}
	var err error
	if head.FC, err = src.Get("mtp.fc.weight", []int{h, 2 * h}); err != nil {
		return nil, err
	}
	if head.PreFCNormEmbedding, err = src.Get("mtp.pre_fc_norm_embedding.weight", []int{h}); err != nil {
		return nil, err
	}
	if head.PreFCNormHidden, err = src.Get("mtp.pre_fc_norm_hidden.weight", []int{h}); err != nil {
		return nil, err
	}
	if head.Norm, err = src.Get("mtp.norm.weight", []int{h}); err != nil {
		return nil, err
	}
	attnShapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	if err != nil {
		return nil, err
	}
	head.Layers = make([]QwenNativeMTPLayer, meta.MTPNumHiddenLayers)
	for i := range head.Layers {
		l := &head.Layers[i]
		prefix := fmt.Sprintf("mtp.layers.%d", i)
		loads := []struct {
			name  string
			dst   **tensor.Tensor
			shape []int
		}{
			{prefix + ".input_layernorm.weight", &l.InputNorm, []int{h}},
			{prefix + ".post_attention_layernorm.weight", &l.PostNorm, []int{h}},
			{prefix + ".self_attn.q_proj.weight", &l.QW, attnShapes.QProj},
			{prefix + ".self_attn.k_proj.weight", &l.KW, attnShapes.KProj},
			{prefix + ".self_attn.v_proj.weight", &l.VW, attnShapes.VProj},
			{prefix + ".self_attn.o_proj.weight", &l.OW, attnShapes.OProj},
			{prefix + ".self_attn.q_norm.weight", &l.QNorm, attnShapes.QNorm},
			{prefix + ".self_attn.k_norm.weight", &l.KNorm, attnShapes.KNorm},
			{prefix + ".mlp.gate_proj.weight", &l.GateW, []int{inter, h}},
			{prefix + ".mlp.up_proj.weight", &l.UpW, []int{inter, h}},
			{prefix + ".mlp.down_proj.weight", &l.DownW, []int{h, inter}},
		}
		for _, load := range loads {
			*load.dst, err = src.Get(load.name, load.shape)
			if err != nil {
				return nil, err
			}
		}
	}
	if err := ValidateQwenNativeMTPHead(head, meta); err != nil {
		return nil, err
	}
	return head, nil
}

func ValidateQwenNativeMTPHead(head *QwenNativeMTPHead, meta loaderconfig.QwenNativeMTPMetadata) error {
	if head == nil {
		return fmt.Errorf("nil Qwen native MTP head")
	}
	if !meta.HasNativeMTP {
		return fmt.Errorf("metadata does not enable native MTP")
	}
	h := meta.HiddenSize
	if h <= 0 {
		return fmt.Errorf("invalid hidden size %d", h)
	}
	if meta.IntermediateSize <= 0 {
		return fmt.Errorf("invalid intermediate size %d", meta.IntermediateSize)
	}
	if err := expectShape(head.FC, []int{h, 2 * h}, "mtp.fc.weight"); err != nil {
		return err
	}
	if err := expectShape(head.PreFCNormEmbedding, []int{h}, "mtp.pre_fc_norm_embedding.weight"); err != nil {
		return err
	}
	if err := expectShape(head.PreFCNormHidden, []int{h}, "mtp.pre_fc_norm_hidden.weight"); err != nil {
		return err
	}
	if err := expectShape(head.Norm, []int{h}, "mtp.norm.weight"); err != nil {
		return err
	}
	if len(head.Layers) != meta.MTPNumHiddenLayers {
		return fmt.Errorf("MTP layer count=%d want %d", len(head.Layers), meta.MTPNumHiddenLayers)
	}
	attnShapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	if err != nil {
		return err
	}
	for i := range head.Layers {
		l := &head.Layers[i]
		prefix := fmt.Sprintf("mtp.layers.%d", i)
		checks := []struct {
			name string
			t    *tensor.Tensor
			want []int
		}{
			{prefix + ".input_layernorm.weight", l.InputNorm, []int{h}},
			{prefix + ".post_attention_layernorm.weight", l.PostNorm, []int{h}},
			{prefix + ".self_attn.q_proj.weight", l.QW, attnShapes.QProj},
			{prefix + ".self_attn.k_proj.weight", l.KW, attnShapes.KProj},
			{prefix + ".self_attn.v_proj.weight", l.VW, attnShapes.VProj},
			{prefix + ".self_attn.o_proj.weight", l.OW, attnShapes.OProj},
			{prefix + ".self_attn.q_norm.weight", l.QNorm, attnShapes.QNorm},
			{prefix + ".self_attn.k_norm.weight", l.KNorm, attnShapes.KNorm},
			{prefix + ".mlp.gate_proj.weight", l.GateW, []int{meta.IntermediateSize, h}},
			{prefix + ".mlp.up_proj.weight", l.UpW, []int{meta.IntermediateSize, h}},
			{prefix + ".mlp.down_proj.weight", l.DownW, []int{h, meta.IntermediateSize}},
		}
		for _, c := range checks {
			if err := expectShape(c.t, c.want, c.name); err != nil {
				return err
			}
		}
	}
	return nil
}

func expectShape(t *tensor.Tensor, want []int, name string) error {
	if t == nil {
		return fmt.Errorf("missing %s", name)
	}
	got := t.Shape()
	if len(got) != len(want) {
		return fmt.Errorf("%s rank=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("%s shape=%v want %v", name, got, want)
		}
	}
	return nil
}
