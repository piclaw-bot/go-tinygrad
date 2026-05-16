package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/backends/simd"
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

type Qwen35LinearAttentionState struct {
	Conv []float32
	SSM  []float32
	Pos  int
}

func NewQwen35LinearAttentionState(meta loaderconfig.QwenNativeMTPMetadata) (Qwen35LinearAttentionState, error) {
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		return Qwen35LinearAttentionState{}, err
	}
	convLen := shapes.ConvDim * meta.LinearConvKernelDim
	ssmLen := meta.LinearNumValueHeads * meta.LinearValueHeadDim * meta.LinearNumKeyHeads * meta.LinearKeyHeadDim
	if convLen < 0 || ssmLen < 0 {
		return Qwen35LinearAttentionState{}, fmt.Errorf("invalid Qwen3.5 linear-attention state dims conv=%d ssm=%d", convLen, ssmLen)
	}
	return Qwen35LinearAttentionState{Conv: make([]float32, convLen), SSM: make([]float32, ssmLen)}, nil
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

type Qwen35BaseModel struct {
	Layers []Qwen35BaseLayer
}

type Qwen35BaseForwardState struct {
	FullK  [][]float32
	FullV  [][]float32
	Linear []Qwen35LinearAttentionState
	Pos    int
}

func NewQwen35BaseForwardState(model *Qwen35BaseModel, meta loaderconfig.QwenNativeMTPMetadata) (Qwen35BaseForwardState, error) {
	if model == nil {
		return Qwen35BaseForwardState{}, fmt.Errorf("nil Qwen3.5 base model")
	}
	state := Qwen35BaseForwardState{
		FullK:  make([][]float32, len(model.Layers)),
		FullV:  make([][]float32, len(model.Layers)),
		Linear: make([]Qwen35LinearAttentionState, len(model.Layers)),
	}
	for i, layer := range model.Layers {
		if layer.Kind == Qwen35LinearAttentionLayerKind {
			linearState, err := NewQwen35LinearAttentionState(meta)
			if err != nil {
				return Qwen35BaseForwardState{}, fmt.Errorf("linear layer %d state: %w", i, err)
			}
			state.Linear[i] = linearState
		}
	}
	return state, nil
}

func (m *Qwen35BaseModel) ForwardOne(input []float32, state Qwen35BaseForwardState, pos int, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, Qwen35BaseForwardState, error) {
	if m == nil {
		return nil, state, fmt.Errorf("nil Qwen3.5 base model")
	}
	if len(state.FullK) != len(m.Layers) || len(state.FullV) != len(m.Layers) || len(state.Linear) != len(m.Layers) {
		return nil, state, fmt.Errorf("Qwen3.5 forward state layer counts K/V/linear=%d/%d/%d want %d", len(state.FullK), len(state.FullV), len(state.Linear), len(m.Layers))
	}
	cur := append([]float32(nil), input...)
	for i := range m.Layers {
		layer := &m.Layers[i]
		switch layer.Kind {
		case Qwen35FullAttentionLayerKind:
			out, curK, curV, err := layer.Full.ForwardWithKV(cur, pos, ropeFreqs, state.FullK[i], state.FullV[i], eps, meta)
			if err != nil {
				return nil, state, fmt.Errorf("Qwen3.5 full-attention layer %d: %w", i, err)
			}
			nextK, nextV, err := appendQwen35FullAttentionKV(state.FullK[i], state.FullV[i], curK, curV, meta)
			if err != nil {
				return nil, state, fmt.Errorf("Qwen3.5 full-attention layer %d cache: %w", i, err)
			}
			state.FullK[i] = nextK
			state.FullV[i] = nextV
			cur = out
		case Qwen35LinearAttentionLayerKind:
			out, nextLinear, err := layer.Linear.ForwardWithState(cur, state.Linear[i], eps, meta)
			if err != nil {
				return nil, state, fmt.Errorf("Qwen3.5 linear-attention layer %d: %w", i, err)
			}
			state.Linear[i] = nextLinear
			cur = out
		default:
			return nil, state, fmt.Errorf("Qwen3.5 layer %d has unsupported kind %q", i, layer.Kind)
		}
	}
	state.Pos = pos + 1
	return cur, state, nil
}

func appendQwen35FullAttentionKV(pastK, pastV, curK, curV []float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, []float32, error) {
	kvDim := meta.NumKeyValueHeads * meta.HeadDim
	if kvDim <= 0 {
		return nil, nil, fmt.Errorf("invalid Qwen3.5 KV dim heads=%d head_dim=%d", meta.NumKeyValueHeads, meta.HeadDim)
	}
	if len(pastK) != len(pastV) {
		return nil, nil, fmt.Errorf("past K/V len mismatch %d/%d", len(pastK), len(pastV))
	}
	if len(curK) != kvDim || len(curV) != kvDim {
		return nil, nil, fmt.Errorf("current K/V len=%d/%d want %d", len(curK), len(curV), kvDim)
	}
	if len(pastK)%kvDim != 0 {
		return nil, nil, fmt.Errorf("past KV len=%d not multiple of %d", len(pastK), kvDim)
	}
	nextK := make([]float32, 0, len(pastK)+len(curK))
	nextK = append(nextK, pastK...)
	nextK = append(nextK, curK...)
	nextV := make([]float32, 0, len(pastV)+len(curV))
	nextV = append(nextV, pastV...)
	nextV = append(nextV, curV...)
	return nextK, nextV, nil
}

func LoadQwen35BaseModelLayers(src Qwen35TensorSource, meta loaderconfig.QwenNativeMTPMetadata) (*Qwen35BaseModel, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Qwen3.5 tensor source")
	}
	count := meta.MainLayerCount()
	if count <= 0 {
		return nil, fmt.Errorf("invalid Qwen3.5 main layer count %d", count)
	}
	out := &Qwen35BaseModel{Layers: make([]Qwen35BaseLayer, count)}
	for i := 0; i < count; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)
		if meta.IsLinearAttentionLayer(i) {
			layer, err := LoadQwen35LinearAttentionLayer(src, meta, prefix)
			if err != nil {
				return nil, fmt.Errorf("load Qwen3.5 linear layer %d: %w", i, err)
			}
			out.Layers[i] = Qwen35BaseLayer{Kind: Qwen35LinearAttentionLayerKind, Linear: layer}
			continue
		}
		layer, err := LoadQwen35FullAttentionLayer(src, meta, prefix)
		if err != nil {
			return nil, fmt.Errorf("load Qwen3.5 full-attention layer %d: %w", i, err)
		}
		out.Layers[i] = Qwen35BaseLayer{Kind: Qwen35FullAttentionLayerKind, Full: layer}
	}
	return out, nil
}

func LoadQwen35FullAttentionLayer(src Qwen35TensorSource, meta loaderconfig.QwenNativeMTPMetadata, prefix string) (*Qwen35FullAttentionLayer, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Qwen3.5 tensor source")
	}
	h := meta.HiddenSize
	inter := meta.IntermediateSize
	shapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	if err != nil {
		return nil, err
	}
	l := &Qwen35FullAttentionLayer{}
	loads := []struct {
		name string
		dst  **tensor.Tensor
		want []int
	}{
		{prefix + ".input_layernorm.weight", &l.InputNorm, []int{h}},
		{prefix + ".post_attention_layernorm.weight", &l.PostNorm, []int{h}},
		{prefix + ".self_attn.q_proj.weight", &l.QW, shapes.QProj},
		{prefix + ".self_attn.k_proj.weight", &l.KW, shapes.KProj},
		{prefix + ".self_attn.v_proj.weight", &l.VW, shapes.VProj},
		{prefix + ".self_attn.o_proj.weight", &l.OW, shapes.OProj},
		{prefix + ".self_attn.q_norm.weight", &l.QNorm, shapes.QNorm},
		{prefix + ".self_attn.k_norm.weight", &l.KNorm, shapes.KNorm},
		{prefix + ".mlp.gate_proj.weight", &l.GateW, []int{inter, h}},
		{prefix + ".mlp.up_proj.weight", &l.UpW, []int{inter, h}},
		{prefix + ".mlp.down_proj.weight", &l.DownW, []int{h, inter}},
	}
	for _, load := range loads {
		*load.dst, err = src.Get(load.name, load.want)
		if err != nil {
			return nil, err
		}
	}
	if err := ValidateQwen35FullAttentionLayer(l, meta, prefix); err != nil {
		return nil, err
	}
	return l, nil
}

func LoadQwen35LinearAttentionLayer(src Qwen35TensorSource, meta loaderconfig.QwenNativeMTPMetadata, prefix string) (*Qwen35LinearAttentionLayer, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Qwen3.5 tensor source")
	}
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		return nil, err
	}
	h := meta.HiddenSize
	inter := meta.IntermediateSize
	l := &Qwen35LinearAttentionLayer{}
	loads := []struct {
		name string
		dst  **tensor.Tensor
		want []int
	}{
		{prefix + ".input_layernorm.weight", &l.InputNorm, []int{h}},
		{prefix + ".post_attention_layernorm.weight", &l.PostNorm, []int{h}},
		{prefix + ".linear_attn.in_proj_qkvz.weight", &l.QKVW, shapes.QKV},
		{prefix + ".linear_attn.in_proj_gate.weight", &l.GateW, shapes.Gate},
		{prefix + ".linear_attn.conv1d.weight", &l.Conv1D, shapes.Conv1D},
		{prefix + ".linear_attn.dt_bias", &l.DTBias, shapes.DTBias},
		{prefix + ".linear_attn.A", &l.A, shapes.A},
		{prefix + ".linear_attn.in_proj_ba.weight", &l.BetaW, shapes.Beta},
		{prefix + ".linear_attn.in_proj_a.weight", &l.AlphaW, shapes.Alpha},
		{prefix + ".linear_attn.norm.weight", &l.Norm, shapes.Norm},
		{prefix + ".linear_attn.out_proj.weight", &l.OutW, shapes.Out},
		{prefix + ".mlp.gate_proj.weight", &l.MLPGateW, []int{inter, h}},
		{prefix + ".mlp.up_proj.weight", &l.MLPUpW, []int{inter, h}},
		{prefix + ".mlp.down_proj.weight", &l.MLPDownW, []int{h, inter}},
	}
	for _, load := range loads {
		*load.dst, err = src.Get(load.name, load.want)
		if err != nil {
			return nil, err
		}
	}
	if err := ValidateQwen35LinearAttentionLayer(l, meta, prefix); err != nil {
		return nil, err
	}
	return l, nil
}

func (l *Qwen35FullAttentionLayer) ForwardWithKV(input []float32, pos int, ropeFreqs, pastK, pastV []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (out, curK, curV []float32, err error) {
	if l == nil {
		return nil, nil, nil, fmt.Errorf("nil Qwen3.5 full-attention layer")
	}
	h := meta.HiddenSize
	if len(input) != h {
		return nil, nil, nil, fmt.Errorf("Qwen3.5 full-attention input len=%d want %d", len(input), h)
	}
	shapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	if err != nil {
		return nil, nil, nil, err
	}
	cur := append([]float32(nil), input...)
	rmsNormInPlace(cur, l.InputNorm.Data(), eps)
	qFull := make([]float32, shapes.QProj[0])
	gemvNT(qFull, cur, l.QW.Data(), h, shapes.QProj[0])
	qDim := shapes.GateSize
	q := append([]float32(nil), qFull[:qDim]...)
	gate := qFull[qDim:]
	k := make([]float32, shapes.KProj[0])
	v := make([]float32, shapes.VProj[0])
	gemvNT(k, cur, l.KW.Data(), h, len(k))
	gemvNT(v, cur, l.VW.Data(), h, len(v))
	normHeads(q, l.QNorm.Data(), meta.NumAttentionHeads, meta.HeadDim, eps)
	normHeads(k, l.KNorm.Data(), meta.NumKeyValueHeads, meta.HeadDim, eps)
	if len(ropeFreqs) > 0 {
		applyRoPE(q, ropeFreqs, pos, meta.NumAttentionHeads, meta.HeadDim)
		applyRoPE(k, ropeFreqs, pos, meta.NumKeyValueHeads, meta.HeadDim)
	}
	curK = append([]float32(nil), k...)
	curV = append([]float32(nil), v...)
	kAll, vAll, err := appendQwenMTPKV(pastK, pastV, curK, curV)
	if err != nil {
		return nil, nil, nil, err
	}
	attn := qwenMTPGroupedAttention(q, kAll, vAll, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
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
	out = make([]float32, h)
	simd.VecAdd(out, resid, down)
	return out, curK, curV, nil
}

func (l *Qwen35FullAttentionLayer) Forward(input []float32, pos int, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, error) {
	out, _, _, err := l.ForwardWithKV(input, pos, ropeFreqs, nil, nil, eps, meta)
	return out, err
}

type Qwen35LinearQKVZ struct {
	Q []float32
	K []float32
	V []float32
	Z []float32
}

func splitQwen35LinearQKVZ(projected []float32, shapes loaderconfig.Qwen35LinearAttentionShapes) (Qwen35LinearQKVZ, error) {
	qLen := shapes.ValueDim
	kLen := shapes.KeyDim
	vLen := shapes.ValueDim
	zLen := shapes.ValueDim
	want := qLen + kLen + vLen + zLen
	if len(projected) != want {
		return Qwen35LinearQKVZ{}, fmt.Errorf("Qwen3.5 linear-attention QKVZ len=%d want %d", len(projected), want)
	}
	off := 0
	out := Qwen35LinearQKVZ{}
	out.Q = append([]float32(nil), projected[off:off+qLen]...)
	off += qLen
	out.K = append([]float32(nil), projected[off:off+kLen]...)
	off += kLen
	out.V = append([]float32(nil), projected[off:off+vLen]...)
	off += vLen
	out.Z = append([]float32(nil), projected[off:off+zLen]...)
	return out, nil
}

func updateQwen35LinearConvState(state []float32, x []float32, kernel int) ([]float32, error) {
	if kernel <= 0 {
		return nil, fmt.Errorf("invalid Qwen3.5 linear-attention conv kernel %d", kernel)
	}
	if len(x) == 0 {
		return nil, fmt.Errorf("empty Qwen3.5 linear-attention conv input")
	}
	want := len(x) * kernel
	if len(state) != want {
		return nil, fmt.Errorf("Qwen3.5 linear-attention conv state len=%d want %d", len(state), want)
	}
	next := make([]float32, len(state))
	if kernel > 1 {
		copy(next[:len(x)*(kernel-1)], state[len(x):])
	}
	copy(next[len(x)*(kernel-1):], x)
	return next, nil
}

func applyQwen35LinearDepthwiseConv(state []float32, weight []float32, convDim, kernel int) ([]float32, error) {
	if convDim <= 0 || kernel <= 0 {
		return nil, fmt.Errorf("invalid Qwen3.5 linear-attention conv dims conv_dim=%d kernel=%d", convDim, kernel)
	}
	if len(state) != convDim*kernel {
		return nil, fmt.Errorf("Qwen3.5 linear-attention conv state len=%d want %d", len(state), convDim*kernel)
	}
	if len(weight) != convDim*kernel {
		return nil, fmt.Errorf("Qwen3.5 linear-attention conv weight len=%d want %d", len(weight), convDim*kernel)
	}
	out := make([]float32, convDim)
	for k := 0; k < kernel; k++ {
		stateOff := k * convDim
		weightOff := k * convDim
		for c := 0; c < convDim; c++ {
			out[c] += state[stateOff+c] * weight[weightOff+c]
		}
	}
	return out, nil
}

func (l *Qwen35LinearAttentionLayer) ForwardWithState(input []float32, state Qwen35LinearAttentionState, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, Qwen35LinearAttentionState, error) {
	if l == nil {
		return nil, state, fmt.Errorf("nil Qwen3.5 linear-attention layer")
	}
	if len(input) != meta.HiddenSize {
		return nil, state, fmt.Errorf("Qwen3.5 linear-attention input len=%d want %d", len(input), meta.HiddenSize)
	}
	want, err := NewQwen35LinearAttentionState(meta)
	if err != nil {
		return nil, state, err
	}
	if len(state.Conv) != len(want.Conv) || len(state.SSM) != len(want.SSM) {
		return nil, state, fmt.Errorf("Qwen3.5 linear-attention state dims conv/ssm=%d/%d want %d/%d", len(state.Conv), len(state.SSM), len(want.Conv), len(want.SSM))
	}
	return nil, state, fmt.Errorf("Qwen3.5 linear-attention forward is not implemented: gated delta-net recurrent update pending")
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
