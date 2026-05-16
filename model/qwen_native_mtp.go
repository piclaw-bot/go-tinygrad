package model

import (
	"fmt"
	"math"

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

type QwenNativeMTPDraftState struct {
	Hidden []float32
	K      []float32
	V      []float32
	Pos    int
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

type QwenNativeMTPPlan struct {
	TokenID  int
	State    QwenNativeMTPDraftState
	MaxSteps int
}

func NewQwenNativeMTPPlan(tokenID int, state QwenNativeMTPDraftState, maxSteps int, meta loaderconfig.QwenNativeMTPMetadata) (QwenNativeMTPPlan, error) {
	if tokenID < 0 {
		return QwenNativeMTPPlan{}, fmt.Errorf("token id %d out of range", tokenID)
	}
	if maxSteps < 0 {
		return QwenNativeMTPPlan{}, fmt.Errorf("max MTP draft steps=%d must be >= 0", maxSteps)
	}
	if !meta.HasNativeMTP {
		return QwenNativeMTPPlan{}, fmt.Errorf("metadata does not enable native MTP")
	}
	if meta.MTPNumHiddenLayers != 1 {
		return QwenNativeMTPPlan{}, fmt.Errorf("native MTP layers=%d, only 1 supported", meta.MTPNumHiddenLayers)
	}
	if len(state.Hidden) != meta.HiddenSize {
		return QwenNativeMTPPlan{}, fmt.Errorf("state hidden len=%d want %d", len(state.Hidden), meta.HiddenSize)
	}
	return QwenNativeMTPPlan{TokenID: tokenID, State: state, MaxSteps: maxSteps}, nil
}

type QwenNativeMTPStats struct {
	DraftedTokens  int `json:"drafted_tokens"`
	AcceptedTokens int `json:"accepted_tokens"`
	BonusTokens    int `json:"bonus_tokens"`
	OutputTokens   int `json:"output_tokens"`
}

func (s QwenNativeMTPStats) Add(other QwenNativeMTPStats) QwenNativeMTPStats {
	s.DraftedTokens += other.DraftedTokens
	s.AcceptedTokens += other.AcceptedTokens
	s.BonusTokens += other.BonusTokens
	s.OutputTokens += other.OutputTokens
	return s
}

func (s QwenNativeMTPStats) Average(n int) QwenNativeMTPStats {
	if n <= 1 {
		return s
	}
	s.DraftedTokens /= n
	s.AcceptedTokens /= n
	s.BonusTokens /= n
	s.OutputTokens /= n
	return s
}

func (s QwenNativeMTPStats) AcceptanceRate() float64 {
	if s.DraftedTokens <= 0 {
		return 0
	}
	return float64(s.AcceptedTokens) / float64(s.DraftedTokens)
}

func QwenNativeMTPStatsFromAcceptance(a MTPAcceptance) QwenNativeMTPStats {
	bonus := 0
	if len(a.OutputTokens) > a.AcceptedPrefixLen {
		bonus = 1
	}
	return QwenNativeMTPStats{
		DraftedTokens:  a.DraftedCount,
		AcceptedTokens: a.AcceptedPrefixLen,
		BonusTokens:    bonus,
		OutputTokens:   len(a.OutputTokens),
	}
}

type QwenNativeMTPStepResult struct {
	InitialState QwenNativeMTPDraftState
	State        QwenNativeMTPDraftState
	StepStates   []QwenNativeMTPDraftState
	Drafted      []int
	Logits       [][]float32
	Acceptance   MTPAcceptance
	Stats        QwenNativeMTPStats
}

func RunQwenNativeMTPPlanFromLogits(head *QwenNativeMTPHead, m *LlamaModel, plan QwenNativeMTPPlan, verifierLogits [][]float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (QwenNativeMTPStepResult, error) {
	if head == nil {
		return QwenNativeMTPStepResult{}, fmt.Errorf("nil Qwen native MTP head")
	}
	_, drafted, logitsRows, stepStates, err := head.DraftStepsDetailed(m, plan.TokenID, plan.State, plan.MaxSteps, eps, meta)
	if err != nil {
		return QwenNativeMTPStepResult{}, err
	}
	if err := ValidateQwenNativeMTPVerifierLogits(m, drafted, verifierLogits); err != nil {
		return QwenNativeMTPStepResult{}, err
	}
	acceptance, err := AcceptMTPDraftFromLogits(drafted, verifierLogits)
	if err != nil {
		return QwenNativeMTPStepResult{}, err
	}
	committed := CommitQwenNativeMTPDraftState(plan.State, stepStates, acceptance)
	stats := QwenNativeMTPStatsFromAcceptance(acceptance)
	return QwenNativeMTPStepResult{InitialState: plan.State, State: committed, StepStates: stepStates, Drafted: drafted, Logits: logitsRows, Acceptance: acceptance, Stats: stats}, nil
}

func RunQwenNativeMTPSpeculativeStepFromLogits(head *QwenNativeMTPHead, m *LlamaModel, tokenID int, state QwenNativeMTPDraftState, verifierLogits [][]float32, maxSteps int, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (QwenNativeMTPStepResult, error) {
	plan, err := NewQwenNativeMTPPlan(tokenID, state, maxSteps, meta)
	if err != nil {
		return QwenNativeMTPStepResult{}, err
	}
	return RunQwenNativeMTPPlanFromLogits(head, m, plan, verifierLogits, eps, meta)
}

func ValidateQwenNativeMTPVerifierLogits(m *LlamaModel, drafted []int, logits [][]float32) error {
	if m == nil {
		return fmt.Errorf("nil main model")
	}
	if m.Config.VocabSize <= 0 {
		return fmt.Errorf("invalid vocab size %d", m.Config.VocabSize)
	}
	if len(logits) != len(drafted)+1 {
		return fmt.Errorf("Qwen native MTP verifier logits rows=%d want drafted+1=%d", len(logits), len(drafted)+1)
	}
	for i, row := range logits {
		if len(row) != m.Config.VocabSize {
			return fmt.Errorf("Qwen native MTP verifier logits row %d len=%d want vocab=%d", i, len(row), m.Config.VocabSize)
		}
	}
	return nil
}

func RunQwenNativeMTPPlan(head *QwenNativeMTPHead, m *LlamaModel, plan QwenNativeMTPPlan, verifierTokens []int, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (QwenNativeMTPStepResult, error) {
	if head == nil {
		return QwenNativeMTPStepResult{}, fmt.Errorf("nil Qwen native MTP head")
	}
	_, drafted, logitsRows, stepStates, err := head.DraftStepsDetailed(m, plan.TokenID, plan.State, plan.MaxSteps, eps, meta)
	if err != nil {
		return QwenNativeMTPStepResult{}, err
	}
	acceptance, err := AcceptMTPDraft(drafted, verifierTokens)
	if err != nil {
		return QwenNativeMTPStepResult{}, err
	}
	committed := CommitQwenNativeMTPDraftState(plan.State, stepStates, acceptance)
	stats := QwenNativeMTPStatsFromAcceptance(acceptance)
	return QwenNativeMTPStepResult{InitialState: plan.State, State: committed, StepStates: stepStates, Drafted: drafted, Logits: logitsRows, Acceptance: acceptance, Stats: stats}, nil
}

func RunQwenNativeMTPSpeculativeStep(head *QwenNativeMTPHead, m *LlamaModel, tokenID int, state QwenNativeMTPDraftState, verifierTokens []int, maxSteps int, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (QwenNativeMTPStepResult, error) {
	plan, err := NewQwenNativeMTPPlan(tokenID, state, maxSteps, meta)
	if err != nil {
		return QwenNativeMTPStepResult{}, err
	}
	return RunQwenNativeMTPPlan(head, m, plan, verifierTokens, eps, meta)
}

func CommitQwenNativeMTPDraftState(initial QwenNativeMTPDraftState, stepStates []QwenNativeMTPDraftState, acceptance MTPAcceptance) QwenNativeMTPDraftState {
	if acceptance.AcceptedPrefixLen <= 0 || len(stepStates) == 0 {
		return initial
	}
	idx := acceptance.AcceptedPrefixLen - 1
	if idx >= len(stepStates) {
		idx = len(stepStates) - 1
	}
	return stepStates[idx]
}

func (head *QwenNativeMTPHead) DraftSteps(m *LlamaModel, tokenID int, state QwenNativeMTPDraftState, maxSteps int, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (QwenNativeMTPDraftState, []int, [][]float32, error) {
	next, tokens, logitsRows, _, err := head.DraftStepsDetailed(m, tokenID, state, maxSteps, eps, meta)
	return next, tokens, logitsRows, err
}

func (head *QwenNativeMTPHead) DraftStepsDetailed(m *LlamaModel, tokenID int, state QwenNativeMTPDraftState, maxSteps int, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (QwenNativeMTPDraftState, []int, [][]float32, []QwenNativeMTPDraftState, error) {
	if maxSteps < 0 {
		return state, nil, nil, nil, fmt.Errorf("max MTP draft steps=%d must be >= 0", maxSteps)
	}
	tokens := make([]int, 0, maxSteps)
	logitsRows := make([][]float32, 0, maxSteps)
	stepStates := make([]QwenNativeMTPDraftState, 0, maxSteps)
	curToken := tokenID
	curState := state
	for i := 0; i < maxSteps; i++ {
		nextState, logits, nextToken, err := head.DraftStep(m, curToken, curState, eps, meta)
		if err != nil {
			return state, nil, nil, nil, fmt.Errorf("MTP draft step %d: %w", i, err)
		}
		tokens = append(tokens, nextToken)
		logitsRows = append(logitsRows, logits)
		stepStates = append(stepStates, nextState)
		curToken = nextToken
		curState = nextState
	}
	return curState, tokens, logitsRows, stepStates, nil
}

func (head *QwenNativeMTPHead) DraftStep(m *LlamaModel, tokenID int, state QwenNativeMTPDraftState, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (QwenNativeMTPDraftState, []float32, int, error) {
	if m == nil {
		return state, nil, 0, fmt.Errorf("nil main model")
	}
	embedding := make([]float32, meta.HiddenSize)
	if err := m.ScaledTokenEmbeddingInto(embedding, tokenID); err != nil {
		return state, nil, 0, err
	}
	cur, err := head.PreProject(embedding, state.Hidden, eps)
	if err != nil {
		return state, nil, 0, err
	}
	if len(head.Layers) == 0 {
		return state, nil, 0, fmt.Errorf("Qwen native MTP head has no layers")
	}
	nextHidden, k, v, err := head.Layers[0].ForwardWithKV(cur, state.Pos, m.RopeFreqs, state.K, state.V, eps, meta)
	if err != nil {
		return state, nil, 0, err
	}
	if head.Norm == nil {
		return state, nil, 0, fmt.Errorf("missing mtp.norm.weight")
	}
	logitHidden := append([]float32(nil), nextHidden...)
	rmsNormInPlace(logitHidden, head.Norm.Data(), eps)
	logits := make([]float32, m.Config.VocabSize)
	if err := m.LMHeadLogitsInto(logits, logitHidden); err != nil {
		return state, nil, 0, err
	}
	token, _, err := ArgmaxLogits(logits)
	if err != nil {
		return state, nil, 0, err
	}
	nextState := QwenNativeMTPDraftState{
		Hidden: nextHidden,
		K:      append(append([]float32(nil), state.K...), k...),
		V:      append(append([]float32(nil), state.V...), v...),
		Pos:    state.Pos + 1,
	}
	return nextState, logits, token, nil
}

func (head *QwenNativeMTPHead) DraftLogits(m *LlamaModel, tokenID int, hidden []float32, pos int, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (nextHidden []float32, logits []float32, token int, err error) {
	nextState, logits, token, err := head.DraftStep(m, tokenID, QwenNativeMTPDraftState{Hidden: hidden, Pos: pos}, eps, meta)
	if err != nil {
		return nil, nil, 0, err
	}
	return nextState.Hidden, logits, token, nil
}

func (head *QwenNativeMTPHead) ForwardOne(embedding, hidden []float32, pos int, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, error) {
	cur, err := head.PreProject(embedding, hidden, eps)
	if err != nil {
		return nil, err
	}
	if len(head.Layers) == 0 {
		return nil, fmt.Errorf("Qwen native MTP head has no layers")
	}
	out, _, _, err := head.Layers[0].ForwardWithKV(cur, pos, ropeFreqs, nil, nil, eps, meta)
	return out, err
}

func (l *QwenNativeMTPLayer) Forward(input []float32, pos int, ropeFreqs []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([]float32, error) {
	out, _, _, err := l.ForwardWithKV(input, pos, ropeFreqs, nil, nil, eps, meta)
	return out, err
}

func (l *QwenNativeMTPLayer) ForwardWithKV(input []float32, pos int, ropeFreqs, pastK, pastV []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) (out, curK, curV []float32, err error) {
	if l == nil {
		return nil, nil, nil, fmt.Errorf("nil Qwen native MTP layer")
	}
	full := Qwen35FullAttentionLayer{
		InputNorm: l.InputNorm,
		PostNorm:  l.PostNorm,
		QW:        l.QW,
		KW:        l.KW,
		VW:        l.VW,
		OW:        l.OW,
		QNorm:     l.QNorm,
		KNorm:     l.KNorm,
		GateW:     l.GateW,
		UpW:       l.UpW,
		DownW:     l.DownW,
	}
	out, curK, curV, err = full.ForwardWithKV(input, pos, ropeFreqs, pastK, pastV, eps, meta)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("Qwen native MTP full-attention layer: %w", err)
	}
	return out, curK, curV, nil
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

func appendQwenMTPKV(pastK, pastV, curK, curV []float32) ([]float32, []float32, error) {
	if len(pastK) != len(pastV) {
		return nil, nil, fmt.Errorf("past MTP KV length mismatch K/V=%d/%d", len(pastK), len(pastV))
	}
	outK := make([]float32, 0, len(pastK)+len(curK))
	outK = append(outK, pastK...)
	outK = append(outK, curK...)
	outV := make([]float32, 0, len(pastV)+len(curV))
	outV = append(outV, pastV...)
	outV = append(outV, curV...)
	return outK, outV, nil
}

func qwenMTPGroupedAttention(q, kAll, vAll []float32, nHeads, nKVHeads, headDim int) []float32 {
	out := make([]float32, nHeads*headDim)
	kvDim := nKVHeads * headDim
	seqLen := 0
	if kvDim > 0 {
		seqLen = len(kAll) / kvDim
	}
	if seqLen <= 0 || len(vAll) < seqLen*kvDim {
		return out
	}
	groups := nHeads / nKVHeads
	if groups <= 0 {
		groups = 1
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores := make([]float32, seqLen)
	for h := 0; h < nHeads; h++ {
		kvh := h / groups
		if kvh >= nKVHeads {
			kvh = nKVHeads - 1
		}
		qBase := h * headDim
		maxScore := float32(math.Inf(-1))
		for t := 0; t < seqLen; t++ {
			kvBase := t*kvDim + kvh*headDim
			var score float32
			for i := 0; i < headDim; i++ {
				score += q[qBase+i] * kAll[kvBase+i]
			}
			score *= scale
			scores[t] = score
			if score > maxScore {
				maxScore = score
			}
		}
		var sum float32
		for t := 0; t < seqLen; t++ {
			scores[t] = float32(math.Exp(float64(scores[t] - maxScore)))
			sum += scores[t]
		}
		if sum == 0 {
			continue
		}
		for t := 0; t < seqLen; t++ {
			w := scores[t] / sum
			vBase := t*kvDim + kvh*headDim
			for i := 0; i < headDim; i++ {
				out[qBase+i] += w * vAll[vBase+i]
			}
		}
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
