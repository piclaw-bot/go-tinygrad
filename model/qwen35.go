package model

import (
	"fmt"
	"math"

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
	QWQ       *Qwen35NVFP4Weight
	KWQ       *Qwen35NVFP4Weight
	VWQ       *Qwen35NVFP4Weight
	OWQ       *Qwen35NVFP4Weight
	GateWQ    *Qwen35NVFP4Weight
	UpWQ      *Qwen35NVFP4Weight
	DownWQ    *Qwen35NVFP4Weight
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
	QKVWQ     *Qwen35NVFP4Weight
	GateWQ    *Qwen35NVFP4Weight
	BetaWQ    *Qwen35NVFP4Weight
	AlphaWQ   *Qwen35NVFP4Weight
	OutWQ     *Qwen35NVFP4Weight
	MLPGateWQ *Qwen35NVFP4Weight
	MLPUpWQ   *Qwen35NVFP4Weight
	MLPDownWQ *Qwen35NVFP4Weight
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

func CloneQwen35LinearAttentionState(state Qwen35LinearAttentionState) Qwen35LinearAttentionState {
	return Qwen35LinearAttentionState{
		Conv: append([]float32(nil), state.Conv...),
		SSM:  append([]float32(nil), state.SSM...),
		Pos:  state.Pos,
	}
}

func CloneQwen35BaseForwardState(state Qwen35BaseForwardState) Qwen35BaseForwardState {
	out := Qwen35BaseForwardState{
		FullK:  make([][]float32, len(state.FullK)),
		FullV:  make([][]float32, len(state.FullV)),
		Linear: make([]Qwen35LinearAttentionState, len(state.Linear)),
		Pos:    state.Pos,
	}
	for i := range state.FullK {
		out.FullK[i] = append([]float32(nil), state.FullK[i]...)
	}
	for i := range state.FullV {
		out.FullV[i] = append([]float32(nil), state.FullV[i]...)
	}
	for i := range state.Linear {
		out.Linear[i] = CloneQwen35LinearAttentionState(state.Linear[i])
	}
	return out
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

func (m *Qwen35BaseModel) ForwardSequence(inputs [][]float32, state Qwen35BaseForwardState, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([][]float32, Qwen35BaseForwardState, error) {
	outs := make([][]float32, 0, len(inputs))
	curState := CloneQwen35BaseForwardState(state)
	for i, input := range inputs {
		out, next, err := m.ForwardOne(input, curState, curState.Pos, ropeFreqs, eps, meta)
		if err != nil {
			return nil, state, fmt.Errorf("Qwen3.5 sequence step %d: %w", i, err)
		}
		outs = append(outs, out)
		curState = next
	}
	return outs, curState, nil
}

func (m *Qwen35BaseModel) ForwardOne(input []float32, state Qwen35BaseForwardState, pos int, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, Qwen35BaseForwardState, error) {
	if m == nil {
		return nil, state, fmt.Errorf("nil Qwen3.5 base model")
	}
	if len(state.FullK) != len(m.Layers) || len(state.FullV) != len(m.Layers) || len(state.Linear) != len(m.Layers) {
		return nil, state, fmt.Errorf("Qwen3.5 forward state layer counts K/V/linear=%d/%d/%d want %d", len(state.FullK), len(state.FullV), len(state.Linear), len(m.Layers))
	}
	state = CloneQwen35BaseForwardState(state)
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
		qdst **Qwen35NVFP4Weight
		want []int
	}{
		{prefix + ".input_layernorm.weight", &l.InputNorm, nil, []int{h}},
		{prefix + ".post_attention_layernorm.weight", &l.PostNorm, nil, []int{h}},
		{prefix + ".self_attn.q_proj.weight", &l.QW, &l.QWQ, shapes.QProj},
		{prefix + ".self_attn.k_proj.weight", &l.KW, &l.KWQ, shapes.KProj},
		{prefix + ".self_attn.v_proj.weight", &l.VW, &l.VWQ, shapes.VProj},
		{prefix + ".self_attn.o_proj.weight", &l.OW, &l.OWQ, shapes.OProj},
		{prefix + ".self_attn.q_norm.weight", &l.QNorm, nil, shapes.QNorm},
		{prefix + ".self_attn.k_norm.weight", &l.KNorm, nil, shapes.KNorm},
		{prefix + ".mlp.gate_proj.weight", &l.GateW, &l.GateWQ, []int{inter, h}},
		{prefix + ".mlp.up_proj.weight", &l.UpW, &l.UpWQ, []int{inter, h}},
		{prefix + ".mlp.down_proj.weight", &l.DownW, &l.DownWQ, []int{h, inter}},
	}
	for _, load := range loads {
		if err := loadQwen35DenseOrNVFP4(src, load.name, load.dst, load.qdst, load.want); err != nil {
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
		qdst **Qwen35NVFP4Weight
		want []int
	}{
		{prefix + ".input_layernorm.weight", &l.InputNorm, nil, []int{h}},
		{prefix + ".post_attention_layernorm.weight", &l.PostNorm, nil, []int{h}},
		{prefix + ".linear_attn.in_proj_qkvz.weight", &l.QKVW, &l.QKVWQ, shapes.QKV},
		{prefix + ".linear_attn.in_proj_gate.weight", &l.GateW, &l.GateWQ, shapes.Gate},
		{prefix + ".linear_attn.conv1d.weight", &l.Conv1D, nil, []int{shapes.ConvDim, 1, meta.LinearConvKernelDim}},
		{prefix + ".linear_attn.dt_bias", &l.DTBias, nil, shapes.DTBias},
		{prefix + ".linear_attn.A", &l.A, nil, shapes.A},
		{prefix + ".linear_attn.in_proj_ba.weight", &l.BetaW, &l.BetaWQ, shapes.Beta},
		{prefix + ".linear_attn.in_proj_a.weight", &l.AlphaW, &l.AlphaWQ, shapes.Alpha},
		{prefix + ".linear_attn.norm.weight", &l.Norm, nil, shapes.Norm},
		{prefix + ".linear_attn.out_proj.weight", &l.OutW, &l.OutWQ, shapes.Out},
		{prefix + ".mlp.gate_proj.weight", &l.MLPGateW, &l.MLPGateWQ, []int{inter, h}},
		{prefix + ".mlp.up_proj.weight", &l.MLPUpW, &l.MLPUpWQ, []int{inter, h}},
		{prefix + ".mlp.down_proj.weight", &l.MLPDownW, &l.MLPDownWQ, []int{h, inter}},
	}
	for _, load := range loads {
		if err := loadQwen35DenseOrNVFP4(src, load.name, load.dst, load.qdst, load.want); err != nil {
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
	if err := qwen35LinearInto(qFull, cur, l.QW, l.QWQ, h, shapes.QProj[0], "q_proj"); err != nil {
		return nil, nil, nil, err
	}
	qDim := shapes.GateSize
	q := append([]float32(nil), qFull[:qDim]...)
	gate := qFull[qDim:]
	k := make([]float32, shapes.KProj[0])
	v := make([]float32, shapes.VProj[0])
	if err := qwen35LinearInto(k, cur, l.KW, l.KWQ, h, len(k), "k_proj"); err != nil {
		return nil, nil, nil, err
	}
	if err := qwen35LinearInto(v, cur, l.VW, l.VWQ, h, len(v), "v_proj"); err != nil {
		return nil, nil, nil, err
	}
	normHeads(q, l.QNorm.Data(), meta.NumAttentionHeads, meta.HeadDim, eps)
	normHeads(k, l.KNorm.Data(), meta.NumKeyValueHeads, meta.HeadDim, eps)
	if len(ropeFreqs) > 0 {
		rotHalf := Qwen35RotaryHalf(meta)
		applyRoPEPartial(q, ropeFreqs, pos, meta.NumAttentionHeads, meta.HeadDim, rotHalf)
		applyRoPEPartial(k, ropeFreqs, pos, meta.NumKeyValueHeads, meta.HeadDim, rotHalf)
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
	if err := qwen35LinearInto(o, attn, l.OW, l.OWQ, len(attn), h, "o_proj"); err != nil {
		return nil, nil, nil, err
	}
	resid := make([]float32, h)
	simd.VecAdd(resid, input, o)
	mlpIn := append([]float32(nil), resid...)
	rmsNormInPlace(mlpIn, l.PostNorm.Data(), eps)
	inter := meta.IntermediateSize
	gateMLP := make([]float32, inter)
	up := make([]float32, inter)
	if err := qwen35LinearInto(gateMLP, mlpIn, l.GateW, l.GateWQ, h, inter, "mlp.gate_proj"); err != nil {
		return nil, nil, nil, err
	}
	if err := qwen35LinearInto(up, mlpIn, l.UpW, l.UpWQ, h, inter, "mlp.up_proj"); err != nil {
		return nil, nil, nil, err
	}
	simd.VecSiLUMul(gateMLP, gateMLP, up)
	down := make([]float32, h)
	if err := qwen35LinearInto(down, gateMLP, l.DownW, l.DownWQ, inter, h, "mlp.down_proj"); err != nil {
		return nil, nil, nil, err
	}
	out = make([]float32, h)
	simd.VecAdd(out, resid, down)
	return out, curK, curV, nil
}

func (l *Qwen35FullAttentionLayer) Forward(input []float32, pos int, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, error) {
	out, _, _, err := l.ForwardWithKV(input, pos, ropeFreqs, nil, nil, eps, meta)
	return out, err
}

type Qwen35LinearQKV struct {
	Q []float32
	K []float32
	V []float32
}

func splitQwen35LinearQKV(projected []float32, shapes loaderconfig.Qwen35LinearAttentionShapes) (Qwen35LinearQKV, error) {
	qLen := shapes.KeyDim
	kLen := shapes.KeyDim
	vLen := shapes.ValueDim
	want := qLen + kLen + vLen
	if len(projected) != want {
		return Qwen35LinearQKV{}, fmt.Errorf("Qwen3.5 linear-attention QKV len=%d want %d", len(projected), want)
	}
	off := 0
	out := Qwen35LinearQKV{}
	out.Q = append([]float32(nil), projected[off:off+qLen]...)
	off += qLen
	out.K = append([]float32(nil), projected[off:off+kLen]...)
	off += kLen
	out.V = append([]float32(nil), projected[off:off+vLen]...)
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

func l2NormalizeInPlace(x []float32, eps float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	scale := float32(1 / math.Sqrt(float64(sum+eps)))
	for i := range x {
		x[i] *= scale
	}
}

func siluInPlace(x []float32) {
	for i := range x {
		x[i] = x[i] * sigmoid(x[i])
	}
}

func applyQwen35LinearDeltaUpdate(ssm, q, k, v, beta, dt, decay []float32, shapes loaderconfig.Qwen35LinearAttentionShapes, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, []float32, error) {
	stateLen := meta.LinearNumValueHeads * meta.LinearValueHeadDim * meta.LinearNumKeyHeads * meta.LinearKeyHeadDim
	if len(ssm) != stateLen {
		return nil, nil, fmt.Errorf("Qwen3.5 linear-attention SSM state len=%d want %d", len(ssm), stateLen)
	}
	if len(q) != shapes.KeyDim || len(k) != shapes.KeyDim || len(v) != shapes.ValueDim {
		return nil, nil, fmt.Errorf("Qwen3.5 linear-attention delta Q/K/V len=%d/%d/%d want %d/%d/%d", len(q), len(k), len(v), shapes.KeyDim, shapes.KeyDim, shapes.ValueDim)
	}
	rank := meta.LinearNumValueHeads
	if len(beta) != rank || len(dt) != rank || len(decay) != rank {
		return nil, nil, fmt.Errorf("Qwen3.5 linear-attention delta rank lens beta/dt/decay=%d/%d/%d want %d", len(beta), len(dt), len(decay), rank)
	}
	if meta.LinearNumValueHeads%meta.LinearNumKeyHeads != 0 {
		return nil, nil, fmt.Errorf("Qwen3.5 linear-attention value heads=%d not divisible by key heads=%d", meta.LinearNumValueHeads, meta.LinearNumKeyHeads)
	}
	next := append([]float32(nil), ssm...)
	out := make([]float32, shapes.ValueDim)
	keyWidth := meta.LinearKeyHeadDim
	for vh := 0; vh < meta.LinearNumValueHeads; vh++ {
		kh := vh % meta.LinearNumKeyHeads
		for vd := 0; vd < meta.LinearValueHeadDim; vd++ {
			vIdx := vh*meta.LinearValueHeadDim + vd
			acc := float32(0)
			for kd := 0; kd < meta.LinearKeyHeadDim; kd++ {
				kIdx := kh*meta.LinearKeyHeadDim + kd
				stateIdx := ((vh*meta.LinearValueHeadDim+vd)*meta.LinearNumKeyHeads+kh)*meta.LinearKeyHeadDim + kd
				next[stateIdx] = next[stateIdx]*decay[vh] + beta[vh]*dt[vh]*v[vIdx]*k[kIdx]
				acc += next[stateIdx] * q[kIdx]
			}
			if keyWidth > 0 {
				out[vIdx] = acc / float32(keyWidth)
			}
		}
	}
	return next, out, nil
}

func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func prepareQwen35LinearDeltaParams(alpha, beta, dtBias, a []float32, rank int) (dt, decay []float32, err error) {
	if rank <= 0 {
		return nil, nil, fmt.Errorf("invalid Qwen3.5 linear-attention rank %d", rank)
	}
	if len(alpha) != rank || len(beta) != rank || len(dtBias) != rank || len(a) != rank {
		return nil, nil, fmt.Errorf("Qwen3.5 linear-attention delta lens alpha/beta/dt_bias/A=%d/%d/%d/%d want %d", len(alpha), len(beta), len(dtBias), len(a), rank)
	}
	dt = make([]float32, rank)
	decay = make([]float32, rank)
	for i := 0; i < rank; i++ {
		dt[i] = softplus(alpha[i] + dtBias[i])
		decay[i] = float32(math.Exp(float64(dt[i] * a[i])))
	}
	return dt, decay, nil
}

func projectQwen35LinearAlphaBeta(input []float32, alphaW, betaW []float32, hidden, rank int) (alpha, beta []float32, err error) {
	if hidden <= 0 || rank <= 0 {
		return nil, nil, fmt.Errorf("invalid Qwen3.5 linear-attention alpha/beta dims hidden=%d rank=%d", hidden, rank)
	}
	if len(input) != hidden {
		return nil, nil, fmt.Errorf("Qwen3.5 linear-attention alpha/beta input len=%d want %d", len(input), hidden)
	}
	want := hidden * rank
	if len(alphaW) != want || len(betaW) != want {
		return nil, nil, fmt.Errorf("Qwen3.5 linear-attention alpha/beta weight lens=%d/%d want %d", len(alphaW), len(betaW), want)
	}
	alpha = make([]float32, rank)
	beta = make([]float32, rank)
	gemvNT(alpha, input, alphaW, hidden, rank)
	gemvNT(beta, input, betaW, hidden, rank)
	return alpha, beta, nil
}

func splitQwen35LinearConvOutput(conv []float32, shapes loaderconfig.Qwen35LinearAttentionShapes) (k, v []float32, err error) {
	if len(conv) != shapes.ConvDim {
		return nil, nil, fmt.Errorf("Qwen3.5 linear-attention conv output len=%d want %d", len(conv), shapes.ConvDim)
	}
	kLen := shapes.ConvDim - shapes.ValueDim
	if kLen <= 0 {
		return nil, nil, fmt.Errorf("invalid Qwen3.5 linear-attention conv split key len=%d", kLen)
	}
	return append([]float32(nil), conv[:kLen]...), append([]float32(nil), conv[kLen:]...), nil
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
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		return nil, state, err
	}
	cur := append([]float32(nil), input...)
	rmsNormInPlace(cur, l.InputNorm.Data(), eps)
	projected := make([]float32, shapes.QKV[1])
	if err := qwen35LinearInto(projected, cur, l.QKVW, l.QKVWQ, meta.HiddenSize, shapes.QKV[1], "linear_attn.in_proj_qkv"); err != nil {
		return nil, state, err
	}
	parts, err := splitQwen35LinearQKV(projected, shapes)
	if err != nil {
		return nil, state, err
	}
	z := make([]float32, shapes.ValueDim)
	if err := qwen35LinearInto(z, cur, l.GateW, l.GateWQ, meta.HiddenSize, shapes.ValueDim, "linear_attn.in_proj_z"); err != nil {
		return nil, state, err
	}
	convInput := make([]float32, 0, shapes.ConvDim)
	convInput = append(convInput, parts.Q...)
	convInput = append(convInput, parts.K...)
	convInput = append(convInput, parts.V...)
	nextConv, err := updateQwen35LinearConvState(state.Conv, convInput, meta.LinearConvKernelDim)
	if err != nil {
		return nil, state, err
	}
	convOut, err := applyQwen35LinearDepthwiseConv(nextConv, l.Conv1D.Data(), shapes.ConvDim, meta.LinearConvKernelDim)
	if err != nil {
		return nil, state, err
	}
	siluInPlace(convOut)
	convParts, err := splitQwen35LinearQKV(convOut, shapes)
	if err != nil {
		return nil, state, err
	}
	l2NormalizeInPlace(convParts.Q, eps)
	l2NormalizeInPlace(convParts.K, eps)
	alpha := make([]float32, meta.LinearNumValueHeads)
	beta := make([]float32, meta.LinearNumValueHeads)
	if err := qwen35LinearInto(alpha, cur, l.AlphaW, l.AlphaWQ, meta.HiddenSize, meta.LinearNumValueHeads, "linear_attn.in_proj_a"); err != nil {
		return nil, state, err
	}
	if err := qwen35LinearInto(beta, cur, l.BetaW, l.BetaWQ, meta.HiddenSize, meta.LinearNumValueHeads, "linear_attn.in_proj_b"); err != nil {
		return nil, state, err
	}
	dt, decay, err := prepareQwen35LinearDeltaParams(alpha, beta, l.DTBias.Data(), l.A.Data(), meta.LinearNumValueHeads)
	if err != nil {
		return nil, state, err
	}
	for i := range beta {
		beta[i] = sigmoid(beta[i])
	}
	nextSSM, deltaOut, err := applyQwen35LinearDeltaUpdate(state.SSM, convParts.Q, convParts.K, convParts.V, beta, dt, decay, shapes, meta)
	if err != nil {
		return nil, state, err
	}
	rmsNormInPlace(deltaOut, l.Norm.Data(), eps)
	for i := range deltaOut {
		deltaOut[i] *= z[i] * sigmoid(z[i])
	}
	projectedOut := make([]float32, meta.HiddenSize)
	if err := qwen35LinearInto(projectedOut, deltaOut, l.OutW, l.OutWQ, shapes.ValueDim, meta.HiddenSize, "linear_attn.out_proj"); err != nil {
		return nil, state, err
	}
	resid := make([]float32, meta.HiddenSize)
	simd.VecAdd(resid, input, projectedOut)
	mlpIn := append([]float32(nil), resid...)
	rmsNormInPlace(mlpIn, l.PostNorm.Data(), eps)
	gateMLP := make([]float32, meta.IntermediateSize)
	up := make([]float32, meta.IntermediateSize)
	if err := qwen35LinearInto(gateMLP, mlpIn, l.MLPGateW, l.MLPGateWQ, meta.HiddenSize, meta.IntermediateSize, "mlp.gate_proj"); err != nil {
		return nil, state, err
	}
	if err := qwen35LinearInto(up, mlpIn, l.MLPUpW, l.MLPUpWQ, meta.HiddenSize, meta.IntermediateSize, "mlp.up_proj"); err != nil {
		return nil, state, err
	}
	simd.VecSiLUMul(gateMLP, gateMLP, up)
	down := make([]float32, meta.HiddenSize)
	if err := qwen35LinearInto(down, gateMLP, l.MLPDownW, l.MLPDownWQ, meta.IntermediateSize, meta.HiddenSize, "mlp.down_proj"); err != nil {
		return nil, state, err
	}
	out := make([]float32, meta.HiddenSize)
	simd.VecAdd(out, resid, down)
	state.Conv = nextConv
	state.SSM = nextSSM
	state.Pos++
	return out, state, nil
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
		if err := expectQwen35DenseOrNVFP4Shape(c.t, qwen35FullQuantForName(l, c.name), c.want, c.name); err != nil {
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
		{prefix + ".linear_attn.conv1d.weight", l.Conv1D, []int{shapes.ConvDim, 1, meta.LinearConvKernelDim}},
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
		if err := expectQwen35DenseOrNVFP4Shape(c.t, qwen35QuantForName(l, c.name), c.want, c.name); err != nil {
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
