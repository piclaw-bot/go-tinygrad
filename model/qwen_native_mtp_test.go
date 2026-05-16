package model

import (
	"strconv"
	"strings"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

type fakeQwenMTPTensorSource map[string]*tensor.Tensor

func (s fakeQwenMTPTensorSource) Get(name string, shape []int) (*tensor.Tensor, error) {
	t := s[name]
	if t == nil {
		return nil, errFakeMissing(name)
	}
	if err := expectShape(t, shape, name); err != nil {
		return nil, err
	}
	return t, nil
}

type errFakeMissing string

func (e errFakeMissing) Error() string { return "missing " + string(e) }

func TestQwenNativeMTPForwardOneSynthetic(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	out, err := head.ForwardOne([]float32{1, 0, 0, 0}, []float32{0, 1, 0, 0}, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardOne: %v", err)
	}
	if len(out) != meta.HiddenSize {
		t.Fatalf("out len=%d want %d", len(out), meta.HiddenSize)
	}
	if _, err := (&QwenNativeMTPHead{}).ForwardOne([]float32{1}, []float32{1}, 1e-6, meta); err == nil {
		t.Fatal("incomplete ForwardOne returned nil error")
	}
}

func TestQwenNativeMTPPreProject(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	// Identity-like FC rows select the normalized embedding stream.
	head.FC = tensor.FromFloat32([]float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
	}, []int{4, 8})
	head.PreFCNormEmbedding = tensor.Ones([]int{4})
	head.PreFCNormHidden = tensor.Ones([]int{4})
	out, err := head.PreProject([]float32{1, 0, 0, 0}, []float32{0, 1, 0, 0}, 1e-6)
	if err != nil {
		t.Fatalf("PreProject: %v", err)
	}
	if len(out) != 4 || out[0] <= 1.9 || out[1] != 0 || out[2] != 0 || out[3] != 0 {
		t.Fatalf("PreProject out=%v", out)
	}
	if _, err := head.PreProject([]float32{1}, []float32{1, 2}, 1e-6); err == nil {
		t.Fatal("bad dims returned nil error")
	}
}

func TestLoadQwenNativeMTPHeadSynthetic(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	src := fakeQwenMTPTensorSourceFromHead(head)
	loaded, err := LoadQwenNativeMTPHead(src, meta)
	if err != nil {
		t.Fatalf("LoadQwenNativeMTPHead: %v", err)
	}
	if err := ValidateQwenNativeMTPHead(loaded, meta); err != nil {
		t.Fatalf("Validate loaded head: %v", err)
	}
	delete(src, "mtp.norm.weight")
	if _, err := LoadQwenNativeMTPHead(src, meta); err == nil || !strings.Contains(err.Error(), "mtp.norm.weight") {
		t.Fatalf("missing tensor err=%v", err)
	}
}

func TestValidateQwenNativeMTPHeadSynthetic(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	if err := ValidateQwenNativeMTPHead(head, meta); err != nil {
		t.Fatalf("ValidateQwenNativeMTPHead: %v", err)
	}
	head.Layers[0].QW = tensor.Zeros([]int{1, 1})
	if err := ValidateQwenNativeMTPHead(head, meta); err == nil || !strings.Contains(err.Error(), "q_proj") {
		t.Fatalf("bad q_proj validation err=%v", err)
	}
}

func testQwenNativeMTPMeta() loaderconfig.QwenNativeMTPMetadata {
	return loaderconfig.QwenNativeMTPMetadata{
		HiddenSize:         4,
		IntermediateSize:   6,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		MTPNumHiddenLayers: 1,
		HasNativeMTP:       true,
	}
}

func fakeQwenMTPTensorSourceFromHead(head *QwenNativeMTPHead) fakeQwenMTPTensorSource {
	src := fakeQwenMTPTensorSource{
		"mtp.fc.weight":                    head.FC,
		"mtp.pre_fc_norm_embedding.weight": head.PreFCNormEmbedding,
		"mtp.pre_fc_norm_hidden.weight":    head.PreFCNormHidden,
		"mtp.norm.weight":                  head.Norm,
	}
	for i, l := range head.Layers {
		prefix := "mtp.layers." + strconv.Itoa(i)
		src[prefix+".input_layernorm.weight"] = l.InputNorm
		src[prefix+".post_attention_layernorm.weight"] = l.PostNorm
		src[prefix+".self_attn.q_proj.weight"] = l.QW
		src[prefix+".self_attn.k_proj.weight"] = l.KW
		src[prefix+".self_attn.v_proj.weight"] = l.VW
		src[prefix+".self_attn.o_proj.weight"] = l.OW
		src[prefix+".self_attn.q_norm.weight"] = l.QNorm
		src[prefix+".self_attn.k_norm.weight"] = l.KNorm
		src[prefix+".mlp.gate_proj.weight"] = l.GateW
		src[prefix+".mlp.up_proj.weight"] = l.UpW
		src[prefix+".mlp.down_proj.weight"] = l.DownW
	}
	return src
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
