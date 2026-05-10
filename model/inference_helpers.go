package model

import (
	"fmt"
	"math"
)

// TokenEmbeddingInto copies the raw token embedding row into dst.
func (m *LlamaModel) TokenEmbeddingInto(dst []float32, tokenID int) error {
	if m == nil || m.EmbedTokens == nil {
		return fmt.Errorf("model embeddings are not loaded")
	}
	h := m.Config.HiddenSize
	if len(dst) != h {
		return fmt.Errorf("token embedding dst len=%d, want %d", len(dst), h)
	}
	if tokenID < 0 || tokenID >= m.Config.VocabSize {
		return fmt.Errorf("token id %d out of range [0,%d)", tokenID, m.Config.VocabSize)
	}
	emb := m.EmbedTokens.Data()
	copy(dst, emb[tokenID*h:(tokenID+1)*h])
	return nil
}

// ScaledTokenEmbeddingInto copies the token embedding row and applies the same
// model-specific decode-time embedding scaling used by Generate.
func (m *LlamaModel) ScaledTokenEmbeddingInto(dst []float32, tokenID int) error {
	if err := m.TokenEmbeddingInto(dst, tokenID); err != nil {
		return err
	}
	if m.Config.ModelType == "gemma3_text" || m.Config.ModelType == "gemma4_text" {
		scale := float32(math.Sqrt(float64(m.Config.HiddenSize)))
		for i := range dst {
			dst[i] = toBF16(dst[i] * scale)
		}
	}
	return nil
}

// LMHeadLogitsInto computes logits = hidden · lm_head^T.
func (m *LlamaModel) LMHeadLogitsInto(logits, hidden []float32) error {
	if m == nil || m.LMHead == nil {
		return fmt.Errorf("model LM head is not loaded")
	}
	h := m.Config.HiddenSize
	vocab := m.Config.VocabSize
	if len(hidden) != h {
		return fmt.Errorf("hidden len=%d, want %d", len(hidden), h)
	}
	if len(logits) != vocab {
		return fmt.Errorf("logits len=%d, want %d", len(logits), vocab)
	}
	lmData := m.LMHead.Data()
	if len(lmData) != vocab*h {
		return fmt.Errorf("LM head data len=%d, want %d", len(lmData), vocab*h)
	}
	for v := 0; v < vocab; v++ {
		row := lmData[v*h : (v+1)*h]
		logits[v] = simdDot(hidden, row)
	}
	return nil
}

// ArgmaxLogits returns the index and value of the maximum logit.
func ArgmaxLogits(logits []float32) (int, float32, error) {
	if len(logits) == 0 {
		return 0, 0, fmt.Errorf("empty logits")
	}
	maxIdx := 0
	maxVal := logits[0]
	for i, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx, maxVal, nil
}
