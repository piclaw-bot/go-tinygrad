package model

import (
	"math"
)

// ForwardLayer runs a single transformer layer on CPU and returns the updated hidden state.
// This is used by the hybrid GPU/CPU forward pass for layers that don't fit in GPU VRAM.
// kvCacheK and kvCacheV are the shared KV caches (same as used by the GPU path).
func (m *LlamaModel) ForwardLayer(hidden []float32, layerIdx, step, pos int, kvCacheK, kvCacheV [][]float32) []float32 {
	cfg := m.Config
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	layer := &m.Layers[layerIdx]

	residual := make([]float32, h)
	copy(residual, hidden)

	// Per-layer dims
	layerHeadDim := cfg.HeadDim
	if layer.HeadDimLocal > 0 {
		layerHeadDim = layer.HeadDimLocal
	}
	qDim := numHeads * layerHeadDim
	layerKVDim := numKVHeads * layerHeadDim

	// RMSNorm
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		rmsNormBF16(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
	} else {
		rmsNormInPlace(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
	}

	// Q projection
	q := make([]float32, qDim)
	if layer.QWm != nil {
		GemvMLQ(q, hidden, layer.QWm)
	} else if layer.QW != nil {
		m.mv(q, hidden, layer.QW.Data(), h, qDim)
	}

	// K, V projections (only for HasKV layers)
	var k, v []float32
	if layer.HasKV {
		k = make([]float32, layerKVDim)
		v = make([]float32, layerKVDim)
		if layer.KWm != nil {
			GemvMLQ(k, hidden, layer.KWm)
			GemvMLQ(v, hidden, layer.VWm)
		} else if layer.KW != nil {
			m.mv(k, hidden, layer.KW.Data(), h, layerKVDim)
			m.mv(v, hidden, layer.VW.Data(), h, layerKVDim)
		}
	}

	// BF16 truncation for Gemma
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		bf16Slice(q)
		if k != nil {
			bf16Slice(k)
			bf16Slice(v)
		}
	}

	// V norm (Gemma4: RMSNormNoScale)
	if cfg.ModelType == "gemma4_text" && v != nil {
		eps := float32(cfg.RMSNormEps)
		for head := 0; head < numKVHeads; head++ {
			sl := v[head*layerHeadDim : (head+1)*layerHeadDim]
			var ss float32
			for _, x := range sl {
				ss += x * x
			}
			scale := float32(1.0 / math.Sqrt(float64(ss/float32(len(sl))+eps)))
			for i := range sl {
				sl[i] *= scale
			}
		}
	}

	// QK-Norm
	normFn := rmsNormInPlace
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		normFn = rmsNormBF16
	}
	if layer.QNorm != nil {
		qNorm := layer.QNorm.Data()
		for head := 0; head < numHeads; head++ {
			normFn(q[head*layerHeadDim:(head+1)*layerHeadDim], qNorm, float32(cfg.RMSNormEps))
		}
		if k != nil {
			kNorm := layer.KNorm.Data()
			for head := 0; head < numKVHeads; head++ {
				normFn(k[head*layerHeadDim:(head+1)*layerHeadDim], kNorm, float32(cfg.RMSNormEps))
			}
		}
	}

	// RoPE
	if cfg.ModelType == "gemma4_text" && m.RopeFreqsSWA != nil {
		isSWA := true
		if len(cfg.LayerTypes) > layerIdx {
			isSWA = cfg.LayerTypes[layerIdx] == "sliding_attention"
		}
		if isSWA {
			applyRoPEPartial(q, m.RopeFreqsSWA, pos, numHeads, layerHeadDim, m.RopeHalfSWA)
			if k != nil {
				applyRoPEPartial(k, m.RopeFreqsSWA, pos, numKVHeads, layerHeadDim, m.RopeHalfSWA)
			}
		} else {
			applyRoPEPartial(q, m.RopeFreqsFull, pos, numHeads, layerHeadDim, m.RopeHalfFull)
			if k != nil {
				applyRoPEPartial(k, m.RopeFreqsFull, pos, numKVHeads, layerHeadDim, m.RopeHalfFull)
			}
		}
	} else {
		applyRoPE(q, m.RopeFreqs, pos, numHeads, layerHeadDim)
		if k != nil {
			applyRoPE(k, m.RopeFreqs, pos, numKVHeads, layerHeadDim)
		}
	}

	// KV cache
	kvLayer := layerIdx
	if !layer.HasKV {
		kvLayer = layer.KVSourceLayer
	}
	if k != nil {
		kvCacheK[kvLayer] = append(kvCacheK[kvLayer], k...)
		kvCacheV[kvLayer] = append(kvCacheV[kvLayer], v...)
	}

	// Attention
	seqLen := pos + 1
	attnSeqLen := seqLen
	attnKVOffset := 0
	if cfg.SlidingWindow > 0 && len(cfg.LayerTypes) > layerIdx && cfg.LayerTypes[layerIdx] == "sliding_attention" {
		if seqLen > cfg.SlidingWindow {
			attnSeqLen = cfg.SlidingWindow
			attnKVOffset = seqLen - cfg.SlidingWindow
		}
	}
	var attnOut []float32
	if cfg.ModelType == "gemma4_text" {
		attnOut = gqaAttentionScale(q, kvCacheK[kvLayer][attnKVOffset*numKVHeads*layerHeadDim:], kvCacheV[kvLayer][attnKVOffset*numKVHeads*layerHeadDim:], attnSeqLen, numHeads, numKVHeads, layerHeadDim, 1.0)
	} else {
		attnOut = gqaAttention(q, kvCacheK[kvLayer][attnKVOffset*numKVHeads*layerHeadDim:], kvCacheV[kvLayer][attnKVOffset*numKVHeads*layerHeadDim:], attnSeqLen, numHeads, numKVHeads, layerHeadDim)
	}

	// Output projection
	oOut := make([]float32, h)
	if layer.OWm != nil {
		GemvMLQ(oOut, attnOut, layer.OWm)
	} else if layer.OW != nil {
		m.mv(oOut, attnOut, layer.OW.Data(), qDim, h)
	}

	// Post-attention norm + residual
	if layer.PreFFNNorm != nil {
		rmsNormInPlace(oOut, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
		for i := range hidden {
			hidden[i] = residual[i] + oOut[i]
		}
		copy(residual, hidden)
	} else {
		for i := range hidden {
			hidden[i] = residual[i] + oOut[i]
		}
		copy(residual, hidden)
		rmsNormInPlace(hidden, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
	}

	// MLP
	mlpInput := hidden
	if layer.PreFFNNorm != nil {
		mlpInput = make([]float32, h)
		copy(mlpInput, hidden)
		if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
			rmsNormBF16(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
		} else {
			rmsNormInPlace(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
		}
	}

	layerInter := cfg.Intermediate
	if layer.GateWm != nil && layer.GateWm.OutDim > 0 {
		layerInter = layer.GateWm.OutDim
	} else if layer.GateW != nil {
		s := layer.GateW.Shape()
		if len(s) >= 2 {
			layerInter = s[1]
		}
	}

	gate := make([]float32, layerInter)
	up := make([]float32, layerInter)
	if layer.GateWm != nil {
		GemvMLQ(gate, mlpInput, layer.GateWm)
		GemvMLQ(up, mlpInput, layer.UpWm)
	} else if layer.GateW != nil {
		m.mv(gate, mlpInput, layer.GateW.Data(), h, layerInter)
		m.mv(up, mlpInput, layer.UpW.Data(), h, layerInter)
	}

	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		bf16Slice(gate)
		bf16Slice(up)
	}

	if cfg.HiddenAct == "gelu_pytorch_tanh" {
		for i := range gate {
			gate[i] = geluTanh(gate[i]) * up[i]
		}
		bf16Slice(gate)
	} else {
		for i := range gate {
			x := gate[i]
			sig := float32(1.0 / (1.0 + math.Exp(float64(-x))))
			gate[i] = x * sig * up[i]
		}
	}

	down := make([]float32, h)
	if layer.DownWm != nil {
		GemvMLQ(down, gate, layer.DownWm)
	} else if layer.DownW != nil {
		m.mv(down, gate, layer.DownW.Data(), layerInter, h)
	}

	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		bf16Slice(down)
	}

	// Post-FFN norm
	if layer.PostFFNNorm != nil {
		if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
			rmsNormBF16(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
		} else {
			rmsNormInPlace(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
		}
	}

	// Residual
	for i := range hidden {
		hidden[i] = residual[i] + down[i]
	}

	// Layer scalar (Gemma4)
	if layer.LayerScalar != 1.0 {
		for i := range hidden {
			hidden[i] *= layer.LayerScalar
		}
	}
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		bf16Slice(hidden)
	}

	return hidden
}
