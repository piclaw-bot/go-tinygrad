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

// Gemma4PerLayerInputs builds Gemma4 per-layer input slices for one token.
// Returned per-layer slices share a single backing buffer and remain valid as
// long as the returned [][]float32 is kept alive.
func (m *LlamaModel) Gemma4PerLayerInputs(hidden []float32, tokenID int) ([][]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	cfg := m.Config
	if m.PerLayerModelProj == nil || cfg.HiddenPerLayer == 0 {
		return nil, nil
	}
	h := cfg.HiddenSize
	hpl := cfg.HiddenPerLayer
	nl := cfg.NumLayers
	totalDim := nl * hpl
	if len(hidden) != h {
		return nil, fmt.Errorf("per-layer input hidden len=%d, want %d", len(hidden), h)
	}
	if tokenID < 0 {
		return nil, fmt.Errorf("token id %d out of range", tokenID)
	}
	if len(m.PerLayerModelProj) != totalDim*h {
		return nil, fmt.Errorf("per-layer model projection len=%d, want %d", len(m.PerLayerModelProj), totalDim*h)
	}
	if len(m.PerLayerProjNorm) != hpl {
		return nil, fmt.Errorf("per-layer projection norm len=%d, want %d", len(m.PerLayerProjNorm), hpl)
	}
	if m.EmbedPerLayer != nil && tokenID < cfg.VocabPerLayer {
		need := cfg.VocabPerLayer * totalDim
		if len(m.EmbedPerLayer) < need {
			return nil, fmt.Errorf("per-layer embedding len=%d, want at least %d", len(m.EmbedPerLayer), need)
		}
	}

	proj := make([]float32, totalDim)
	gemvNT(proj, hidden, m.PerLayerModelProj, h, totalDim)
	for i := range proj {
		proj[i] *= m.PerLayerProjScale
	}
	for l := 0; l < nl; l++ {
		sl := proj[l*hpl : (l+1)*hpl]
		rmsNormInPlace(sl, m.PerLayerProjNorm, float32(cfg.RMSNormEps))
	}
	if m.EmbedPerLayer != nil && tokenID < cfg.VocabPerLayer {
		embRow := m.EmbedPerLayer[tokenID*totalDim : (tokenID+1)*totalDim]
		for i := range proj {
			proj[i] = (proj[i] + embRow[i]*m.EmbedPerLayerScale) * m.PerLayerInputScale
		}
	}

	perLayerInputs := make([][]float32, nl)
	for l := 0; l < nl; l++ {
		perLayerInputs[l] = proj[l*hpl : (l+1)*hpl]
	}
	return perLayerInputs, nil
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
