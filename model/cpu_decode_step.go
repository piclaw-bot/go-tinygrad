package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/backends/simd"
)

// FinishCPUDecodeStep applies the model's final decode norm, computes LM-head
// logits, and returns the greedy token. It mutates hidden in the same way the
// historical Generate path did, and returns a copy of that final activation so
// callers can retain it independently of scratch buffers.
func (m *LlamaModel) FinishCPUDecodeStep(hidden []float32) (finalActivation []float32, logits []float32, token int, err error) {
	if m == nil {
		return nil, nil, 0, fmt.Errorf("nil model")
	}
	cfg := m.Config
	if cfg.HiddenSize <= 0 || cfg.VocabSize <= 0 {
		return nil, nil, 0, fmt.Errorf("invalid decode dims hidden=%d vocab=%d", cfg.HiddenSize, cfg.VocabSize)
	}
	if len(hidden) != cfg.HiddenSize {
		return nil, nil, 0, fmt.Errorf("hidden len=%d, want %d", len(hidden), cfg.HiddenSize)
	}
	if m.Norm == nil {
		return nil, nil, 0, fmt.Errorf("model final norm is not loaded")
	}
	norm := m.Norm.Data()
	if len(norm) < cfg.HiddenSize {
		return nil, nil, 0, fmt.Errorf("final norm len=%d, want at least %d", len(norm), cfg.HiddenSize)
	}
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		simd.RMSNormBF16(hidden, norm, float32(cfg.RMSNormEps))
	} else {
		rmsNormInPlace(hidden, norm, float32(cfg.RMSNormEps))
	}
	logits = make([]float32, cfg.VocabSize)
	if err := m.LMHeadLogitsInto(logits, hidden); err != nil {
		return nil, nil, 0, err
	}
	token, _, err = ArgmaxLogits(logits)
	if err != nil {
		return nil, nil, 0, err
	}
	return append([]float32(nil), hidden...), logits, token, nil
}

// finishCPUDecodeStep is kept as the internal spelling used by existing decode
// paths; external orchestration layers should call FinishCPUDecodeStep.
func (m *LlamaModel) finishCPUDecodeStep(hidden []float32) (finalActivation []float32, logits []float32, token int, err error) {
	return m.FinishCPUDecodeStep(hidden)
}
