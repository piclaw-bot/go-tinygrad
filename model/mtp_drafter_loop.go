package model

import "fmt"

// MTPDrafterState carries the hidden-state-conditioned drafter inputs between
// speculative iterations. PreviousToken is the last token emitted/accepted by
// the verifier path; Activation is the main-model-width verifier activation
// used together with that token's embedding for the next drafter step.
type MTPDrafterState struct {
	PreviousToken int
	Activation    []float32
}

// MTPDrafterStepResult is one drafter iteration: the drafted token plus the
// projected main-model-width activation to carry into the next drafter step.
type MTPDrafterStepResult struct {
	Token          int
	Logits         []float32
	NextActivation []float32
	NextState      MTPDrafterState
}

// MTPDrafterExternalKV is the read-only main-model KV view consumed by q-only
// drafter layers. SourceLayers maps drafter layer index -> main-model KV layer.
// K/V are flat per-source-layer sequences with element width equal to the
// drafter layer's KV head count times head dim.
type MTPDrafterExternalKV struct {
	K            [][]float32
	V            [][]float32
	SourceLayers []int
	SeqLen       int
}

// NewMTPDrafterState validates and copies the activation carry for one drafter
// loop. The activation width is the main/backbone model hidden size, not the
// assistant hidden size.
func NewMTPDrafterState(previousToken int, activation []float32, backboneHiddenSize int) (MTPDrafterState, error) {
	if previousToken < 0 {
		return MTPDrafterState{}, fmt.Errorf("previous token %d out of range", previousToken)
	}
	if backboneHiddenSize <= 0 {
		return MTPDrafterState{}, fmt.Errorf("invalid backbone hidden size %d", backboneHiddenSize)
	}
	if len(activation) != backboneHiddenSize {
		return MTPDrafterState{}, fmt.Errorf("activation len=%d, want %d", len(activation), backboneHiddenSize)
	}
	return MTPDrafterState{PreviousToken: previousToken, Activation: append([]float32(nil), activation...)}, nil
}

// RunMTPDrafterStep is one hidden-state-conditioned drafter iteration. The
// complete q-only attention loop is still pending; this function implements the
// projection/LM-head shell and supports zero-layer synthetic drafter fixtures for
// runtime validation. Real drafter assets with q-only layers still return an
// explicit not-implemented error until external-KV attention is wired.
func (m *LlamaModel) RunMTPDrafterStep(d *Gemma4MTPDrafter, state MTPDrafterState) (MTPDrafterStepResult, error) {
	return m.RunMTPDrafterStepWithExternalKV(d, state, nil)
}

// RunMTPDrafterStepWithExternalKV is RunMTPDrafterStep plus the explicit
// external/main-model KV view needed by q-only drafter layers.
func (m *LlamaModel) RunMTPDrafterStepWithExternalKV(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV) (MTPDrafterStepResult, error) {
	if err := m.validateMTPDrafterStepModel(d, state, externalKV); err != nil {
		return MTPDrafterStepResult{}, err
	}
	backboneEmbedding := make([]float32, d.BackboneHiddenSize)
	if err := m.TokenEmbeddingInto(backboneEmbedding, state.PreviousToken); err != nil {
		return MTPDrafterStepResult{}, fmt.Errorf("drafter backbone embedding: %w", err)
	}
	assistantHidden := make([]float32, d.Config.HiddenSize)
	if err := d.PreProjectInto(assistantHidden, backboneEmbedding, state.Activation); err != nil {
		return MTPDrafterStepResult{}, err
	}
	for l := 0; l < d.Config.NumLayers; l++ {
		var err error
		assistantHidden, err = runMTPDrafterQOnlyLayer(d, assistantHidden, l, externalKV)
		if err != nil {
			return MTPDrafterStepResult{}, err
		}
	}
	if d.Norm != nil {
		norm := d.Norm.Data()
		if len(norm) < d.Config.HiddenSize {
			return MTPDrafterStepResult{}, fmt.Errorf("drafter final norm len=%d, want at least %d", len(norm), d.Config.HiddenSize)
		}
		drafterRMSNormInPlace(d, assistantHidden, norm)
	}
	nextActivation := make([]float32, d.BackboneHiddenSize)
	if err := d.PostProjectInto(nextActivation, assistantHidden); err != nil {
		return MTPDrafterStepResult{}, err
	}
	logits := make([]float32, m.Config.VocabSize)
	if err := m.LMHeadLogitsInto(logits, nextActivation); err != nil {
		return MTPDrafterStepResult{}, err
	}
	tok, _, err := ArgmaxLogits(logits)
	if err != nil {
		return MTPDrafterStepResult{}, err
	}
	nextState, err := NewMTPDrafterState(tok, nextActivation, d.BackboneHiddenSize)
	if err != nil {
		return MTPDrafterStepResult{}, err
	}
	return MTPDrafterStepResult{Token: tok, Logits: logits, NextActivation: append([]float32(nil), nextActivation...), NextState: nextState}, nil
}

func (m *LlamaModel) validateMTPDrafterStepModel(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	if d == nil {
		return fmt.Errorf("nil drafter")
	}
	if d.Config.HiddenSize <= 0 || d.BackboneHiddenSize <= 0 || d.Config.VocabSize <= 0 {
		return fmt.Errorf("invalid drafter dims hidden=%d backbone=%d vocab=%d", d.Config.HiddenSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	if m.Config.HiddenSize != d.BackboneHiddenSize || m.Config.VocabSize != d.Config.VocabSize {
		return fmt.Errorf("model/drafter dims mismatch model h/vocab=%d/%d drafter backbone/vocab=%d/%d", m.Config.HiddenSize, m.Config.VocabSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	if state.PreviousToken < 0 || state.PreviousToken >= d.Config.VocabSize {
		return fmt.Errorf("previous token %d out of range [0,%d)", state.PreviousToken, d.Config.VocabSize)
	}
	if len(state.Activation) != d.BackboneHiddenSize {
		return fmt.Errorf("state activation len=%d, want %d", len(state.Activation), d.BackboneHiddenSize)
	}
	if len(d.PreProjection) == 0 || len(d.PostProjection) == 0 {
		return fmt.Errorf("drafter projection weights are not loaded")
	}
	if d.Config.NumLayers != len(d.Layers) {
		return fmt.Errorf("drafter layer count=%d, want %d", len(d.Layers), d.Config.NumLayers)
	}
	if d.Config.NumLayers == 0 {
		return nil
	}
	if d.Norm == nil || len(d.Norm.Data()) < d.Config.HiddenSize {
		return fmt.Errorf("drafter final norm is not loaded or too small")
	}
	return validateMTPDrafterExternalKV(d, externalKV)
}

func runMTPDrafterQOnlyLayer(d *Gemma4MTPDrafter, hidden []float32, layerIdx int, externalKV *MTPDrafterExternalKV) ([]float32, error) {
	if d == nil || layerIdx < 0 || layerIdx >= len(d.Layers) {
		return nil, fmt.Errorf("invalid drafter layer %d", layerIdx)
	}
	layer := &d.Layers[layerIdx]
	h := d.Config.HiddenSize
	headDim := d.Config.HeadDim
	if layer.HeadDimLocal > 0 {
		headDim = layer.HeadDimLocal
	}
	qDim := d.Config.NumHeads * headDim
	source := externalKV.SourceLayers[layerIdx]
	residual := append([]float32(nil), hidden...)
	normed := append([]float32(nil), hidden...)
	drafterRMSNormInPlace(d, normed, layer.InputNorm.Data())
	q := make([]float32, qDim)
	gemvNT(q, normed, layer.QW, h, qDim)
	qNorm := layer.QNorm.Data()
	for head := 0; head < d.Config.NumHeads; head++ {
		drafterRMSNormInPlace(d, q[head*headDim:(head+1)*headDim], qNorm)
	}
	attnOut := gqaAttention(q, externalKV.K[source], externalKV.V[source], externalKV.SeqLen, d.Config.NumHeads, d.Config.NumKVHeads, headDim)
	if attnOut == nil {
		return nil, fmt.Errorf("drafter layer %d external attention failed", layerIdx)
	}
	oOut := make([]float32, h)
	gemvNT(oOut, attnOut, layer.OW, qDim, h)
	if layer.PreFFNNorm != nil {
		drafterRMSNormInPlace(d, oOut, layer.PostNorm.Data())
		for i := 0; i < h; i++ {
			hidden[i] = residual[i] + oOut[i]
		}
		copy(residual, hidden)
	} else {
		for i := 0; i < h; i++ {
			hidden[i] = residual[i] + oOut[i]
		}
		copy(residual, hidden)
		drafterRMSNormInPlace(d, hidden, layer.PostNorm.Data())
	}
	mlpInput := hidden
	if layer.PreFFNNorm != nil {
		mlpInput = append([]float32(nil), hidden...)
		drafterRMSNormInPlace(d, mlpInput, layer.PreFFNNorm.Data())
	}
	gate := make([]float32, d.Config.Intermediate)
	up := make([]float32, d.Config.Intermediate)
	gemvNT(gate, mlpInput, layer.GateW, h, d.Config.Intermediate)
	gemvNT(up, mlpInput, layer.UpW, h, d.Config.Intermediate)
	for i := range gate {
		gate[i] = geluTanh(gate[i]) * up[i]
	}
	down := make([]float32, h)
	gemvNT(down, gate, layer.DownW, d.Config.Intermediate, h)
	if layer.PostFFNNorm != nil {
		drafterRMSNormInPlace(d, down, layer.PostFFNNorm.Data())
	}
	for i := 0; i < h; i++ {
		hidden[i] = residual[i] + down[i]
	}
	if layer.LayerScalar != 1 {
		for i := range hidden {
			hidden[i] *= layer.LayerScalar
		}
	}
	return hidden, nil
}

func drafterRMSNormInPlace(d *Gemma4MTPDrafter, x, weight []float32) {
	if d != nil && (d.Config.ModelType == "gemma3_text" || d.Config.ModelType == "gemma4_text") {
		rmsNormBF16(x, weight, float32(d.Config.RMSNormEps))
		return
	}
	rmsNormInPlace(x, weight, float32(d.Config.RMSNormEps))
}

func validateMTPDrafterExternalKV(d *Gemma4MTPDrafter, externalKV *MTPDrafterExternalKV) error {
	if externalKV == nil {
		return fmt.Errorf("MTP drafter external KV is required for q-only layers")
	}
	if externalKV.SeqLen <= 0 {
		return fmt.Errorf("invalid MTP drafter external KV seq len %d", externalKV.SeqLen)
	}
	if len(externalKV.SourceLayers) != d.Config.NumLayers {
		return fmt.Errorf("drafter external KV source layers=%d, want %d", len(externalKV.SourceLayers), d.Config.NumLayers)
	}
	if d.Config.NumHeads <= 0 || d.Config.NumKVHeads <= 0 || d.Config.HeadDim <= 0 || d.Config.Intermediate <= 0 {
		return fmt.Errorf("invalid drafter q-only dims heads=%d kvHeads=%d headDim=%d intermediate=%d", d.Config.NumHeads, d.Config.NumKVHeads, d.Config.HeadDim, d.Config.Intermediate)
	}
	for i := range d.Layers {
		layer := &d.Layers[i]
		headDim := d.Config.HeadDim
		if layer.HeadDimLocal > 0 {
			headDim = layer.HeadDimLocal
		}
		qDim, ok := checkedProduct(d.Config.NumHeads, headDim)
		if headDim <= 0 || !ok {
			return fmt.Errorf("invalid drafter layer %d q dim heads=%d headDim=%d", i, d.Config.NumHeads, headDim)
		}
		kvDim, ok := checkedProduct(d.Config.NumKVHeads, headDim)
		if !ok {
			return fmt.Errorf("invalid drafter layer %d KV dim kvHeads=%d headDim=%d", i, d.Config.NumKVHeads, headDim)
		}
		if layer.KVSourceLayer != -1 {
			return fmt.Errorf("drafter layer %d has unexpected owned/shared KV source %d, want q-only -1", i, layer.KVSourceLayer)
		}
		source := externalKV.SourceLayers[i]
		if source < 0 || source >= len(externalKV.K) || source >= len(externalKV.V) {
			return fmt.Errorf("drafter layer %d external KV source %d out of range K/V=%d/%d", i, source, len(externalKV.K), len(externalKV.V))
		}
		wantKV, ok := checkedProduct(externalKV.SeqLen, kvDim)
		if !ok {
			return fmt.Errorf("drafter layer %d external KV length overflows seq=%d kvDim=%d", i, externalKV.SeqLen, kvDim)
		}
		if len(externalKV.K[source]) != wantKV || len(externalKV.V[source]) != wantKV {
			return fmt.Errorf("drafter layer %d external KV K/V=%d/%d, want %d", i, len(externalKV.K[source]), len(externalKV.V[source]), wantKV)
		}
		if layer.InputNorm == nil || layer.PostNorm == nil || layer.PreFFNNorm == nil || layer.PostFFNNorm == nil || layer.QNorm == nil {
			return fmt.Errorf("drafter layer %d missing q-only norms", i)
		}
		if len(layer.InputNorm.Data()) < d.Config.HiddenSize || len(layer.PostNorm.Data()) < d.Config.HiddenSize || len(layer.PreFFNNorm.Data()) < d.Config.HiddenSize || len(layer.PostFFNNorm.Data()) < d.Config.HiddenSize || len(layer.QNorm.Data()) < headDim {
			return fmt.Errorf("drafter layer %d norm dims are too small", i)
		}
		if len(layer.QW) != qDim*d.Config.HiddenSize || len(layer.OW) != d.Config.HiddenSize*qDim {
			return fmt.Errorf("drafter layer %d attention weight dims Q/O=%d/%d, want %d/%d", i, len(layer.QW), len(layer.OW), qDim*d.Config.HiddenSize, d.Config.HiddenSize*qDim)
		}
		if len(layer.GateW) != d.Config.Intermediate*d.Config.HiddenSize || len(layer.UpW) != d.Config.Intermediate*d.Config.HiddenSize || len(layer.DownW) != d.Config.HiddenSize*d.Config.Intermediate {
			return fmt.Errorf("drafter layer %d MLP weight dims are invalid", i)
		}
	}
	return nil
}
