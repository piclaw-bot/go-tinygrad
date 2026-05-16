package model

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/simd"
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

func (head *QwenNativeMTPHead) DraftLogits(m *LlamaModel, tokenID int, hidden []float32, pos int, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (nextHidden []float32, logits []float32, token int, err error) {
	if m == nil {
		return nil, nil, 0, fmt.Errorf("nil main model")
	}
	embedding := make([]float32, meta.HiddenSize)
	if err := m.ScaledTokenEmbeddingInto(embedding, tokenID); err != nil {
		return nil, nil, 0, err
	}
	nextHidden, err = head.ForwardOne(embedding, hidden, pos, m.RopeFreqs, eps, meta)
	if err != nil {
		return nil, nil, 0, err
	}
	if head.Norm == nil {
		return nil, nil, 0, fmt.Errorf("missing mtp.norm.weight")
	}
	logitHidden := append([]float32(nil), nextHidden...)
	rmsNormInPlace(logitHidden, head.Norm.Data(), eps)
	logits = make([]float32, m.Config.VocabSize)
	if err := m.LMHeadLogitsInto(logits, logitHidden); err != nil {
		return nil, nil, 0, err
	}
	token, _, err = ArgmaxLogits(logits)
	if err != nil {
		return nil, nil, 0, err
	}
	return nextHidden, logits, token, nil
}

func (head *QwenNativeMTPHead) ForwardOne(embedding, hidden []float32, pos int, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, error) {
	cur, err := head.PreProject(embedding, hidden, eps)
	if err != nil {
		return nil, err
	}
	if len(head.Layers) == 0 {
		return nil, fmt.Errorf("Qwen native MTP head has no layers")
	}
	return head.Layers[0].Forward(cur, pos, ropeFreqs, eps, meta)
}

func (l *QwenNativeMTPLayer) Forward(input []float32, pos int, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, error) {
	if l == nil {
		return nil, fmt.Errorf("nil Qwen native MTP layer")
	}
	h := meta.HiddenSize
	if len(input) != h {
		return nil, fmt.Errorf("MTP layer input len=%d want %d", len(input), h)
	}
	attnShapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	if err != nil {
		return nil, err
	}
	cur := append([]float32(nil), input...)
	rmsNormInPlace(cur, l.InputNorm.Data(), eps)
	qFull := make([]float32, attnShapes.QProj[0])
	gemvNT(qFull, cur, l.QW.Data(), h, attnShapes.QProj[0])
	qDim := attnShapes.GateSize
	q := append([]float32(nil), qFull[:qDim]...)
	gate := qFull[qDim:]
	k := make([]float32, attnShapes.KProj[0])
	v := make([]float32, attnShapes.VProj[0])
	gemvNT(k, cur, l.KW.Data(), h, len(k))
	gemvNT(v, cur, l.VW.Data(), h, len(v))
	normHeads(q, l.QNorm.Data(), meta.NumAttentionHeads, meta.HeadDim, eps)
	normHeads(k, l.KNorm.Data(), meta.NumKeyValueHeads, meta.HeadDim, eps)
	if len(ropeFreqs) > 0 {
		applyRoPE(q, ropeFreqs, pos, meta.NumAttentionHeads, meta.HeadDim)
		applyRoPE(k, ropeFreqs, pos, meta.NumKeyValueHeads, meta.HeadDim)
	}
	attn := singleTokenGroupedAttention(q, k, v, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	for i := range attn {
		attn[i] *= sigmoid(gate[i])
	}
	o := make([]float32, h)
	gemvNT(o, attn, l.OW.Data(), len(attn), h)
	resid := make([]float32, h)
	simd.VecAdd(resid, input, o)
	mlpIn := append([]float32(nil), resid...)
	rmsNormInPlace(mlpIn, l.PostNorm.Data(), eps)
	inter := meta.IntermediateSize
	gateMLP := make([]float32, inter)
	up := make([]float32, inter)
	gemvNT(gateMLP, mlpIn, l.GateW.Data(), h, inter)
	gemvNT(up, mlpIn, l.UpW.Data(), h, inter)
	simd.VecSiLUMul(gateMLP, gateMLP, up)
	down := make([]float32, h)
	gemvNT(down, gateMLP, l.DownW.Data(), inter, h)
	out := make([]float32, h)
	simd.VecAdd(out, resid, down)
	return out, nil
}

func (head *QwenNativeMTPHead) PreProject(embedding, hidden []float32, eps float32) ([]float32, error) {
	if head == nil {
		return nil, fmt.Errorf("nil Qwen native MTP head")
	}
	if head.FC == nil || head.PreFCNormEmbedding == nil || head.PreFCNormHidden == nil {
		return nil, fmt.Errorf("incomplete Qwen native MTP preprojection tensors")
	}
	h := len(embedding)
	if h <= 0 || len(hidden) != h {
		return nil, fmt.Errorf("embedding/hidden dims=%d/%d", len(embedding), len(hidden))
	}
	if len(head.PreFCNormEmbedding.Data()) < h || len(head.PreFCNormHidden.Data()) < h {
		return nil, fmt.Errorf("preprojection norm dims too small")
	}
	fcShape := head.FC.Shape()
	if len(fcShape) != 2 || fcShape[0] != h || fcShape[1] != 2*h {
		return nil, fmt.Errorf("mtp.fc shape=%v want [%d %d]", fcShape, h, 2*h)
	}
	e := append([]float32(nil), embedding...)
	hh := append([]float32(nil), hidden...)
	rmsNormInPlace(e, head.PreFCNormEmbedding.Data(), eps)
	rmsNormInPlace(hh, head.PreFCNormHidden.Data(), eps)
	concat := make([]float32, 0, 2*h)
	concat = append(concat, e...)
	concat = append(concat, hh...)
	out := make([]float32, h)
	gemvNT(out, concat, head.FC.Data(), 2*h, h)
	return out, nil
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

func normHeads(x, weight []float32, nHeads, headDim int, eps float32) {
	for h := 0; h < nHeads; h++ {
		start := h * headDim
		rmsNormInPlace(x[start:start+headDim], weight, eps)
	}
}

func singleTokenGroupedAttention(q, k, v []float32, nHeads, nKVHeads, headDim int) []float32 {
	out := make([]float32, nHeads*headDim)
	groups := nHeads / nKVHeads
	if groups <= 0 {
		groups = 1
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for h := 0; h < nHeads; h++ {
		kvh := h / groups
		if kvh >= nKVHeads {
			kvh = nKVHeads - 1
		}
		qBase := h * headDim
		kvBase := kvh * headDim
		var score float32
		for i := 0; i < headDim; i++ {
			score += q[qBase+i] * k[kvBase+i]
		}
		// Sequence length is one in the first skeleton, so softmax(score)=1.
		_ = score * scale
		copy(out[qBase:qBase+headDim], v[kvBase:kvBase+headDim])
	}
	return out
}

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
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
