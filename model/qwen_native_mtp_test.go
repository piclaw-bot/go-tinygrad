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

func TestQwenNativeMTPLayerForwardWithKV(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	input, err := head.PreProject([]float32{1, 0, 0, 0}, []float32{0, 1, 0, 0}, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	out, k, v, err := head.Layers[0].ForwardWithKV(input, 0, nil, nil, nil, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardWithKV first: %v", err)
	}
	if len(out) != meta.HiddenSize || len(k) != meta.NumKeyValueHeads*meta.HeadDim || len(v) != len(k) {
		t.Fatalf("lens out/k/v=%d/%d/%d", len(out), len(k), len(v))
	}
	out2, _, _, err := head.Layers[0].ForwardWithKV(input, 1, nil, k, v, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardWithKV history: %v", err)
	}
	if len(out2) != meta.HiddenSize {
		t.Fatalf("out2 len=%d", len(out2))
	}
	if _, _, _, err := head.Layers[0].ForwardWithKV(input, 1, nil, k, nil, 1e-6, meta); err == nil {
		t.Fatal("mismatched past KV returned nil error")
	}
}

func TestQwenNativeMTPForwardOneAcceptsRoPE(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	rope := make([]float32, meta.HeadDim*2)
	for i := 0; i < meta.HeadDim; i++ {
		rope[i*2] = 1
		rope[i*2+1] = 0
	}
	out, err := head.ForwardOne([]float32{1, 0, 0, 0}, []float32{0, 1, 0, 0}, 0, rope, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardOne with RoPE: %v", err)
	}
	if len(out) != meta.HiddenSize {
		t.Fatalf("out len=%d", len(out))
	}
}

func TestQwenNativeMTPStatsFromAcceptance(t *testing.T) {
	acc, err := AcceptMTPDraft([]int{1, 2, 3}, []int{1, 2, 9, 8})
	if err != nil {
		t.Fatal(err)
	}
	stats := QwenNativeMTPStatsFromAcceptance(acc)
	if stats.DraftedTokens != 3 || stats.AcceptedTokens != 2 || stats.BonusTokens != 1 || stats.OutputTokens != 3 || stats.AcceptanceRate() != float64(2)/3 {
		t.Fatalf("stats=%+v rate=%f", stats, stats.AcceptanceRate())
	}
	sum := stats.Add(QwenNativeMTPStats{DraftedTokens: 3, AcceptedTokens: 1, BonusTokens: 1, OutputTokens: 2})
	if sum.DraftedTokens != 6 || sum.AcceptedTokens != 3 || sum.BonusTokens != 2 || sum.OutputTokens != 5 {
		t.Fatalf("sum=%+v", sum)
	}
	avg := sum.Average(2)
	if avg.DraftedTokens != 3 || avg.AcceptedTokens != 1 || avg.BonusTokens != 1 || avg.OutputTokens != 2 {
		t.Fatalf("avg=%+v", avg)
	}
	if (QwenNativeMTPStats{}).AcceptanceRate() != 0 {
		t.Fatal("zero stats acceptance rate not zero")
	}
}

func TestCommitQwenNativeMTPDraftState(t *testing.T) {
	initial := QwenNativeMTPDraftState{Pos: 10}
	states := []QwenNativeMTPDraftState{{Pos: 11}, {Pos: 12}, {Pos: 13}}
	acc, err := AcceptMTPDraft([]int{1, 2, 3}, []int{1, 9, 8, 7})
	if err != nil {
		t.Fatal(err)
	}
	if got := CommitQwenNativeMTPDraftState(initial, states, acc); got.Pos != 11 {
		t.Fatalf("commit rejected Pos=%d want 11", got.Pos)
	}
	acc, err = AcceptMTPDraft([]int{1, 2, 3}, []int{1, 2, 3, 7})
	if err != nil {
		t.Fatal(err)
	}
	if got := CommitQwenNativeMTPDraftState(initial, states, acc); got.Pos != 13 {
		t.Fatalf("commit all accepted Pos=%d want 13", got.Pos)
	}
}

func TestRunQwenNativeMTPSpeculativeStepSynthetic(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	m := syntheticQwenMTPMainModel(meta)
	state := QwenNativeMTPDraftState{Hidden: []float32{0, 1, 0, 0}}
	_, drafted, _, err := head.DraftSteps(m, 0, state, 2, 1e-6, meta)
	if err != nil {
		t.Fatalf("DraftSteps seed: %v", err)
	}
	verifier := append([]int(nil), drafted...)
	verifier = append(verifier, 1) // bonus token
	res, err := RunQwenNativeMTPSpeculativeStep(head, m, 0, state, verifier, 2, 1e-6, meta)
	if err != nil {
		t.Fatalf("RunQwenNativeMTPSpeculativeStep: %v", err)
	}
	if res.Acceptance.AcceptedPrefixLen != len(res.Drafted) || len(res.Acceptance.OutputTokens) != len(res.Drafted)+1 {
		t.Fatalf("acceptance=%+v drafted=%v", res.Acceptance, res.Drafted)
	}
	if len(res.StepStates) != len(res.Drafted) || res.State.Pos != len(res.Drafted) {
		t.Fatalf("state=%+v stepStates=%d drafted=%d", res.State, len(res.StepStates), len(res.Drafted))
	}
	if res.Stats.DraftedTokens != len(res.Drafted) || res.Stats.AcceptedTokens != len(res.Drafted) || res.Stats.OutputTokens != len(res.Drafted)+1 {
		t.Fatalf("stats=%+v drafted=%v", res.Stats, res.Drafted)
	}
	if _, err := RunQwenNativeMTPSpeculativeStep(nil, m, 0, state, verifier, 2, 1e-6, meta); err == nil {
		t.Fatal("nil head returned nil error")
	}
}

func TestQwenNativeMTPDraftSteps(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	m := syntheticQwenMTPMainModel(meta)
	state := QwenNativeMTPDraftState{Hidden: []float32{0, 1, 0, 0}}
	next, tokens, logitsRows, err := head.DraftSteps(m, 0, state, 2, 1e-6, meta)
	if err != nil {
		t.Fatalf("DraftSteps: %v", err)
	}
	if len(tokens) != 2 || len(logitsRows) != 2 || next.Pos != 2 {
		t.Fatalf("tokens=%v logitsRows=%d next=%+v", tokens, len(logitsRows), next)
	}
	if _, _, _, err := head.DraftSteps(m, 0, state, -1, 1e-6, meta); err == nil {
		t.Fatal("negative DraftSteps returned nil error")
	}
}

func TestQwenNativeMTPDraftStepState(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	m := syntheticQwenMTPMainModel(meta)
	state := QwenNativeMTPDraftState{Hidden: []float32{0, 1, 0, 0}}
	state, _, tok, err := head.DraftStep(m, 0, state, 1e-6, meta)
	if err != nil {
		t.Fatalf("DraftStep first: %v", err)
	}
	if state.Pos != 1 || len(state.K) != meta.NumKeyValueHeads*meta.HeadDim || len(state.V) != len(state.K) || tok < 0 {
		t.Fatalf("state=%+v tok=%d", state, tok)
	}
	state, _, _, err = head.DraftStep(m, tok, state, 1e-6, meta)
	if err != nil {
		t.Fatalf("DraftStep second: %v", err)
	}
	if state.Pos != 2 || len(state.K) != 2*meta.NumKeyValueHeads*meta.HeadDim || len(state.V) != len(state.K) {
		t.Fatalf("state after second=%+v", state)
	}
}

func TestQwenNativeMTPDraftLogitsSynthetic(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	m := syntheticQwenMTPMainModel(meta)
	_, logits, tok, err := head.DraftLogits(m, 0, []float32{0, 1, 0, 0}, 0, 1e-6, meta)
	if err != nil {
		t.Fatalf("DraftLogits: %v", err)
	}
	if len(logits) != 2 || tok < 0 || tok >= 2 {
		t.Fatalf("logits=%v tok=%d", logits, tok)
	}
	if _, _, _, err := head.DraftLogits(nil, 0, nil, 0, 1e-6, meta); err == nil {
		t.Fatal("nil model DraftLogits returned nil error")
	}
}

func TestQwenNativeMTPForwardOneSynthetic(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	out, err := head.ForwardOne([]float32{1, 0, 0, 0}, []float32{0, 1, 0, 0}, 0, nil, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardOne: %v", err)
	}
	if len(out) != meta.HiddenSize {
		t.Fatalf("out len=%d want %d", len(out), meta.HiddenSize)
	}
	if _, err := (&QwenNativeMTPHead{}).ForwardOne([]float32{1}, []float32{1}, 0, nil, 1e-6, meta); err == nil {
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

func syntheticQwenMTPMainModel(meta loaderconfig.QwenNativeMTPMetadata) *LlamaModel {
	return &LlamaModel{
		Config: LlamaConfig{VocabSize: 2, HiddenSize: meta.HiddenSize, RMSNormEps: 1e-6},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		}, []int{2, 4}),
		LMHead: tensor.FromFloat32([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		}, []int{2, 4}),
	}
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
