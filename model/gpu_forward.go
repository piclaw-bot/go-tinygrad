package model

// GPU-resident LLM forward pass using DevBuf.
// tinygrad approach: all weights + hidden state on GPU.
// Every op dispatches to GPU kernel with CPU fallback.

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/rcarmo/go-pherence/gpu"
	"github.com/rcarmo/go-pherence/tensor"
)

// GPUModel wraps a LlamaModel with GPU-resident weights and buffers.
type GPUModel struct {
	CPU    *LlamaModel
	Config LlamaConfig

	Layers []gpuLayerBufs

	// Work buffers (GPU-resident)
	hidden, residual, normed *gpu.DevBuf
	q, k, v, attnOut, oOut   *gpu.DevBuf
	gate, up, down           *gpu.DevBuf
	// Gemma4 per-layer input gating buffers
	perLayerProjBuf   *gpu.DevBuf // [numLayers * hiddenPerLayer]
	perLayerEmbedBuf  *gpu.DevBuf // scratch row upload, same shape as perLayerProjBuf
	pliGateBuf        *gpu.DevBuf // [hiddenPerLayer]
	pliProjBuf        *gpu.DevBuf // [hidden]
	perLayerModelProj *gpu.DevBuf // [numLayers*hiddenPerLayer, hidden]
	perLayerProjNorm  *gpu.DevBuf // [hiddenPerLayer]

	// KV cache (GPU-resident for fast path, CPU for fallback)
	kvCacheK, kvCacheV [][]float32   // CPU slices
	kvGPU_K, kvGPU_V   []*gpu.DevBuf // GPU buffers [maxSeq * kvDim] per layer

	// RoPE precomputed cos/sin
	ropeCosSin     *gpu.DevBuf
	ropeCosSinSWA  *gpu.DevBuf // Gemma4: SWA RoPE
	ropeCosSinFull *gpu.DevBuf // Gemma4: full attention RoPE
	ropeHalfSWA    int
	ropeHalfFull   int

	// Final norm + lm_head stay on CPU (vocab is huge)
	normWeight []float32

	// GPU LM head
	lmHeadGPU *gpu.DevBuf // [vocab × h] F32 on GPU
	normGPU   *gpu.DevBuf // final norm weights on GPU
	logitsGPU *gpu.DevBuf // [vocab] logits output on GPU
	lmHead    []float32   // [vocab, h]
	vocabSize int
}

type gpuLayerBufs struct {
	QW, KW, VW, OW          *gpu.DevBuf
	QB, KB, VB              *gpu.DevBuf
	QNorm, KNorm            *gpu.DevBuf // QK-Norm (Qwen3)
	GateW, UpW, DownW       *gpu.DevBuf
	InputNorm, PostNorm     *gpu.DevBuf
	PreFFNNorm, PostFFNNorm *gpu.DevBuf // Gemma3/4
	// GPTQ quantized (CPU fallback)
	QWq, KWq, VWq, OWq   *QuantWeight
	GateWq, UpWq, DownWq *QuantWeight
	// GPTQ on GPU
	QWg, KWg, VWg, OWg   *gpu.GPUQuantWeight
	GateWg, UpWg, DownWg *gpu.GPUQuantWeight
	// MLX on GPU
	QWmg, KWmg, VWmg, OWmg  *gpu.GPUMLXWeight
	GateWmg, UpWmg, DownWmg *gpu.GPUMLXWeight
	// MLX on CPU
	QWm, KWm, VWm, OWm   *MLXQuantWeight
	GateWm, UpWm, DownWm *MLXQuantWeight
	// Gemma4 per-layer input gating on GPU (raw row-major F32 weights)
	PLIGate, PLIProj, PLIPostNorm *gpu.DevBuf
}

// LoadGPUModel uploads model weights to GPU using DevBuf.
func LoadGPUModel(m *LlamaModel) (*GPUModel, error) {
	runtime.LockOSThread()
	start := time.Now()

	cfg := m.Config
	h := cfg.HiddenSize
	kvDim := cfg.HeadDim * cfg.NumKVHeads
	inter := cfg.Intermediate

	g := &GPUModel{
		CPU:       m,
		Config:    cfg,
		vocabSize: cfg.VocabSize,
	}

	// Work buffers — sized for max across all layer types
	maxHeadDim := cfg.HeadDim
	if cfg.GlobalHeadDim > maxHeadDim {
		maxHeadDim = cfg.GlobalHeadDim
	}
	maxQDim := cfg.NumHeads * maxHeadDim
	maxKVDim := cfg.NumKVHeads * maxHeadDim
	// Max intermediate (Gemma4 double-wide MLP for shared layers)
	maxInter := inter
	for _, layer := range m.Layers {
		if layer.GateWm != nil && layer.GateWm.OutDim > maxInter {
			maxInter = layer.GateWm.OutDim
		}
		if layer.GateW != nil {
			s := layer.GateW.Shape()
			if len(s) > 0 && s[0] > maxInter {
				maxInter = s[0]
			}
		}
	}

	g.hidden = gpu.NewDevBuf(h)
	g.residual = gpu.NewDevBuf(h)
	g.normed = gpu.NewDevBuf(h)
	g.q = gpu.NewDevBuf(maxQDim)
	g.k = gpu.NewDevBuf(maxKVDim)
	g.v = gpu.NewDevBuf(maxKVDim)
	g.attnOut = gpu.NewDevBuf(maxQDim)
	g.oOut = gpu.NewDevBuf(maxQDim)
	g.gate = gpu.NewDevBuf(maxInter)
	g.up = gpu.NewDevBuf(maxInter)
	g.down = gpu.NewDevBuf(h)

	// Gemma4 per-layer work buffers
	if cfg.ModelType == "gemma4_text" && cfg.HiddenPerLayer > 0 {
		totalDim := cfg.NumLayers * cfg.HiddenPerLayer
		g.perLayerProjBuf = gpu.NewDevBuf(totalDim)
		g.perLayerEmbedBuf = gpu.NewDevBuf(totalDim)
		g.pliGateBuf = gpu.NewDevBuf(cfg.HiddenPerLayer)
		g.pliProjBuf = gpu.NewDevBuf(h)
	}

	// Try to move work buffers to GPU
	useGPU := true
	for _, buf := range []*gpu.DevBuf{g.hidden, g.residual, g.normed, g.q, g.k, g.v, g.attnOut, g.oOut, g.gate, g.up, g.down, g.perLayerProjBuf, g.perLayerEmbedBuf, g.pliGateBuf, g.pliProjBuf} {
		if buf == nil {
			continue
		}
		if err := buf.ToGPU(); err != nil {
			useGPU = false
			break
		}
	}

	// Upload per-layer weights
	wrapTensor := func(t *tensor.Tensor) *gpu.DevBuf {
		if t == nil {
			return nil
		}
		b := gpu.NewDevBufFrom(t.Data())
		if useGPU {
			b.ToGPU()
		}
		return b
	}
	wrapSlice := func(x []float32) *gpu.DevBuf {
		if x == nil {
			return nil
		}
		b := gpu.NewDevBufFrom(x)
		if useGPU {
			b.ToGPU()
		}
		return b
	}

	g.Layers = make([]gpuLayerBufs, len(m.Layers))
	gpu.InitAllKernels()
	for i, layer := range m.Layers {
		gl := gpuLayerBufs{
			InputNorm: wrapTensor(layer.InputNorm),
			PostNorm:  wrapTensor(layer.PostNorm),
		}

		if layer.QWq != nil {
			gl.QWq = layer.QWq
			gl.KWq = layer.KWq
			gl.VWq = layer.VWq
			gl.OWq = layer.OWq
			gl.GateWq = layer.GateWq
			gl.UpWq = layer.UpWq
			gl.DownWq = layer.DownWq
			gl.QWg, _ = gpu.UploadQuantWeight(layer.QWq.QWeight, layer.QWq.GIdx, layer.QWq.Scales, layer.QWq.InDim, layer.QWq.OutDim)
			if gpu.Q4Ready() {
				gl.KWg, _ = gpu.UploadQuantWeight(layer.KWq.QWeight, layer.KWq.GIdx, layer.KWq.Scales, layer.KWq.InDim, layer.KWq.OutDim)
				gl.VWg, _ = gpu.UploadQuantWeight(layer.VWq.QWeight, layer.VWq.GIdx, layer.VWq.Scales, layer.VWq.InDim, layer.VWq.OutDim)
				gl.OWg, _ = gpu.UploadQuantWeight(layer.OWq.QWeight, layer.OWq.GIdx, layer.OWq.Scales, layer.OWq.InDim, layer.OWq.OutDim)
				gl.GateWg, _ = gpu.UploadQuantWeight(layer.GateWq.QWeight, layer.GateWq.GIdx, layer.GateWq.Scales, layer.GateWq.InDim, layer.GateWq.OutDim)
				gl.UpWg, _ = gpu.UploadQuantWeight(layer.UpWq.QWeight, layer.UpWq.GIdx, layer.UpWq.Scales, layer.UpWq.InDim, layer.UpWq.OutDim)
				gl.DownWg, _ = gpu.UploadQuantWeight(layer.DownWq.QWeight, layer.DownWq.GIdx, layer.DownWq.Scales, layer.DownWq.InDim, layer.DownWq.OutDim)
			}
		} else if layer.QWm != nil {
			// MLX quantized: upload to GPU
			gl.QWm = layer.QWm
			gl.KWm = layer.KWm
			gl.VWm = layer.VWm
			gl.OWm = layer.OWm
			gl.GateWm = layer.GateWm
			gl.UpWm = layer.UpWm
			gl.DownWm = layer.DownWm
			if gpu.SgemmReady() {
				um := func(qw *MLXQuantWeight) *gpu.GPUMLXWeight {
					w, err := gpu.UploadMLXWeight(qw.Weight, qw.Scales, qw.Biases, qw.InDim, qw.OutDim, qw.GroupSize)
					if err != nil && i == 0 {
						fmt.Printf("[gpu] MLX upload %dx%d: %v\n", qw.OutDim, qw.InDim, err)
					}
					return w
				}
				gl.QWmg = um(layer.QWm)
				gl.KWmg = um(layer.KWm)
				gl.VWmg = um(layer.VWm)
				gl.OWmg = um(layer.OWm)
				gl.GateWmg = um(layer.GateWm)
				gl.UpWmg = um(layer.UpWm)
				gl.DownWmg = um(layer.DownWm)
			}
		} else {
			gl.QW = wrapTensor(layer.QW)
			gl.KW = wrapTensor(layer.KW)
			gl.VW = wrapTensor(layer.VW)
			gl.OW = wrapTensor(layer.OW)
			gl.GateW = wrapTensor(layer.GateW)
			gl.UpW = wrapTensor(layer.UpW)
			gl.DownW = wrapTensor(layer.DownW)
		}

		gl.PreFFNNorm = wrapTensor(layer.PreFFNNorm)
		gl.PostFFNNorm = wrapTensor(layer.PostFFNNorm)
		gl.QB = wrapTensor(layer.QB)
		gl.KB = wrapTensor(layer.KB)
		gl.VB = wrapTensor(layer.VB)
		gl.QNorm = wrapTensor(layer.QNorm)
		gl.KNorm = wrapTensor(layer.KNorm)
		gl.PLIGate = wrapSlice(layer.PLIGate)
		gl.PLIProj = wrapSlice(layer.PLIProj)
		gl.PLIPostNorm = wrapSlice(layer.PLIPostNorm)

		g.Layers[i] = gl
	}

	// Gemma4 model-level per-layer input gating weights
	g.perLayerModelProj = wrapSlice(m.PerLayerModelProj)
	g.perLayerProjNorm = wrapSlice(m.PerLayerProjNorm)

	// KV cache (per-layer kvDim for Gemma4)
	g.kvCacheK = make([][]float32, len(m.Layers))
	g.kvCacheV = make([][]float32, len(m.Layers))
	for l := range g.kvCacheK {
		lkv := kvDim
		if m.Layers[l].HeadDimLocal > 0 {
			lkv = cfg.NumKVHeads * m.Layers[l].HeadDimLocal
		}
		g.kvCacheK[l] = make([]float32, 0, 2048*lkv)
		g.kvCacheV[l] = make([]float32, 0, 2048*lkv)
	}

	// Final layers stay CPU
	g.normWeight = m.Norm.Data()
	if m.LMHead != nil {
		g.lmHead = m.LMHead.Data()
	} else {
		g.lmHead = m.EmbedTokens.Data()
	}

	// Upload final norm + logits buffer to GPU (small, before weights)
	if useGPU && gpu.SgemmReady() {
		g.normGPU = gpu.NewDevBuf(len(g.normWeight))
		copy(g.normGPU.Data(), g.normWeight)
		g.normGPU.MarkDirty()
		g.normGPU.ToGPU()

		g.logitsGPU = gpu.NewDevBuf(cfg.VocabSize)
		g.logitsGPU.ToGPU()
	}

	// Gemma4: precompute dual RoPE tables
	if cfg.ModelType == "gemma4_text" && m.RopeFreqsSWA != nil {
		g.ropeHalfSWA = m.RopeHalfSWA
		g.ropeHalfFull = m.RopeHalfFull
		// Upload SWA table
		g.ropeCosSinSWA = gpu.NewDevBufFrom(m.RopeFreqsSWA)
		g.ropeCosSinSWA.ToGPU()
		// Upload Full table
		g.ropeCosSinFull = gpu.NewDevBufFrom(m.RopeFreqsFull)
		g.ropeCosSinFull.ToGPU()
	}

	device := "CPU"
	// Precompute RoPE cos/sin table
	{
		headDimL := cfg.HeadDim
		halfDim := headDimL / 2
		maxSeqL := 2048
		csData := make([]float32, maxSeqL*headDimL)
		for p := 0; p < maxSeqL; p++ {
			for i := 0; i < halfDim; i++ {
				freq := m.RopeFreqs[p*halfDim+i]
				csData[p*headDimL+i*2] = float32(math.Cos(float64(freq)))
				csData[p*headDimL+i*2+1] = float32(math.Sin(float64(freq)))
			}
		}
		g.ropeCosSin = gpu.NewDevBufFrom(csData)
		g.ropeCosSin.ToGPU()
	}

	// GPU KV cache: per-layer kvDim
	maxSeq := 2048
	g.kvGPU_K = make([]*gpu.DevBuf, len(m.Layers))
	g.kvGPU_V = make([]*gpu.DevBuf, len(m.Layers))
	for i := range g.kvGPU_K {
		lkv := kvDim
		if m.Layers[i].HeadDimLocal > 0 {
			lkv = cfg.NumKVHeads * m.Layers[i].HeadDimLocal
		}
		g.kvGPU_K[i] = gpu.NewDevBuf(maxSeq * lkv)
		g.kvGPU_V[i] = gpu.NewDevBuf(maxSeq * lkv)
		g.kvGPU_K[i].ToGPU()
		g.kvGPU_V[i].ToGPU()
	}

	// Upload LM head to GPU — may need to split if VRAM is limited
	if useGPU && gpu.SgemmReady() {
		free, _ := gpu.MemInfo()
		lmBytes := uint64(len(g.lmHead) * 4)
		if free > lmBytes+64*1024*1024 { // need LM head + 64MB headroom
			g.lmHeadGPU = gpu.NewDevBuf(len(g.lmHead))
			copy(g.lmHeadGPU.Data(), g.lmHead)
			g.lmHeadGPU.MarkDirty()
			if err := g.lmHeadGPU.ToGPU(); err == nil {
				fmt.Printf("[model] LM head on GPU (%.0f MB)\n", float64(lmBytes)/1e6)
			} else {
				g.lmHeadGPU = nil
			}
		} else {
			fmt.Printf("[model] LM head stays on CPU (need %.0f MB, free %.0f MB)\n", float64(lmBytes)/1e6, float64(free)/1e6)
		}
	}

	elapsed := time.Since(start)
	if useGPU {
		device = "GPU"
	}
	fmt.Printf("[model] Weights on %s (%d layers, %v)\n", device, len(g.Layers), elapsed.Round(time.Millisecond))

	return g, nil
}

func (g *GPUModel) gemv(out, x, W *gpu.DevBuf, inDim, outDim int) {
	if g.CPU.Large {
		gpu.DevGemv(out, x, W, outDim, inDim) // W is [outDim, inDim]
	} else {
		gpu.DevGemvNN(out, x, W, inDim, outDim) // W is [inDim, outDim] (pre-transposed)
	}
}

// Generate produces tokens with GPU-resident forward pass.
func (g *GPUModel) Generate(tokenIDs []int, maxTokens int) []int {
	runtime.LockOSThread()
	cfg := g.Config
	// Prepend BOS token if model requires it (Gemma)
	if cfg.BOSTokenID > 0 && (cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text") {
		tokenIDs = append([]int{cfg.BOSTokenID}, tokenIDs...)
	}
	if cfg.ModelType == "gemma4_text" && g.CPU != nil && g.CPU.Tok != nil {
		turnStart, turnEnd := -1, -1
		newlineID := -1
		for id, tok := range g.CPU.Tok.InvVocab {
			if tok == "<|turn>" {
				turnStart = id
			}
			if tok == "<turn|>" {
				turnEnd = id
			}
			if tok == "\n" {
				newlineID = id
			}
		}
		if turnStart >= 0 && turnEnd >= 0 && newlineID >= 0 {
			user := g.CPU.Tok.Encode("user")
			mdl := g.CPU.Tok.Encode("model")
			wrapped := []int{cfg.BOSTokenID, turnStart}
			wrapped = append(wrapped, user...)
			wrapped = append(wrapped, newlineID)
			wrapped = append(wrapped, tokenIDs[1:]...)
			wrapped = append(wrapped, turnEnd)
			wrapped = append(wrapped, newlineID)
			wrapped = append(wrapped, turnStart)
			wrapped = append(wrapped, mdl...)
			wrapped = append(wrapped, newlineID)
			tokenIDs = wrapped
		}
	}
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	_ = headDim * numKVHeads
	inter := cfg.Intermediate
	m := g.CPU

	output := make([]int, len(tokenIDs), len(tokenIDs)+maxTokens)
	copy(output, tokenIDs)

	// Temp CPU buffers for RoPE + attention (sequential ops)
	var kd, vd []float32
	logits := make([]float32, g.vocabSize)

	// Batched prefill: process all prompt tokens at once
	prefillStart := 0
	if len(tokenIDs) > 1 && gpu.BatchGEMMReady() && cfg.ModelType != "gemma4_text" {
		if lastHidden := g.prefillGPU(tokenIDs); lastHidden != nil {
			// Prefill succeeded — skip to decode phase
			prefillStart = len(tokenIDs) - 1 // skip all but last prompt token
			// Set up hidden state from prefill result
			copy(g.hidden.Data(), lastHidden)
			g.hidden.MarkDirty()
		}
	}

	for step := prefillStart; step < len(tokenIDs)+maxTokens-1; step++ {
		var tokID int
		if step < len(tokenIDs) {
			tokID = tokenIDs[step]
		} else {
			tokID = output[len(output)-1]
		}
		pos := step

		skipLayers := (step == prefillStart && prefillStart > 0)
		var hd []float32

		if !skipLayers {
			// Embedding (CPU — vocab too large for VRAM on small GPUs)
			embData := m.EmbedTokens.Data()
			hd = g.hidden.Data()
			copy(hd, embData[tokID*h:(tokID+1)*h])
			g.hidden.MarkDirty()
			// Gemma3/4: scale embeddings by sqrt(hidden_size)
			if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
				scale := float32(math.Sqrt(float64(h)))
				for i := range hd {
					hd[i] *= scale
				}
				g.hidden.MarkDirty()
				if cfg.ModelType == "gemma4_text" {
					gpu.DevToBF16(g.hidden, h)
				}
			}

			// Gemma4: per-layer input gating (GPU path with CPU fallback)
			var perLayerInputs [][]float32
			usePLIGPU := cfg.ModelType == "gemma4_text" && g.perLayerModelProj != nil && g.perLayerProjNorm != nil && g.perLayerProjBuf != nil && g.perLayerProjBuf.GPUPtr() != nil
			if cfg.ModelType == "gemma4_text" && m.PerLayerModelProj != nil && cfg.HiddenPerLayer > 0 {
				hpl := cfg.HiddenPerLayer
				nl := cfg.NumLayers
				totalDim := nl * hpl
				if usePLIGPU {
					gpu.DevGemv(g.perLayerProjBuf, g.hidden, g.perLayerModelProj, totalDim, h)
					gpu.DevScale(g.perLayerProjBuf, g.perLayerProjBuf, m.PerLayerProjScale)
					for ll := 0; ll < nl; ll++ {
						sl := g.perLayerProjBuf.Slice(ll*hpl, hpl)
						gpu.DevRMSNorm(sl, sl, g.perLayerProjNorm, float32(cfg.RMSNormEps))
					}
					if m.EmbedPerLayer != nil && tokID < cfg.VocabPerLayer {
						embRow := m.EmbedPerLayer[tokID*totalDim : (tokID+1)*totalDim]
						copy(g.perLayerEmbedBuf.Data(), embRow)
						g.perLayerEmbedBuf.MarkDirty()
						gpu.DevScale(g.perLayerEmbedBuf, g.perLayerEmbedBuf, m.EmbedPerLayerScale)
						gpu.DevAdd(g.perLayerProjBuf, g.perLayerProjBuf, g.perLayerEmbedBuf)
						gpu.DevScale(g.perLayerProjBuf, g.perLayerProjBuf, m.PerLayerInputScale)
					}
				} else {
					proj := make([]float32, totalDim)
					hd2 := g.hidden.Data()
					gemvNT(proj, hd2, m.PerLayerModelProj, h, totalDim)
					for i := range proj {
						proj[i] *= m.PerLayerProjScale
					}
					for ll := 0; ll < nl; ll++ {
						sl := proj[ll*hpl : (ll+1)*hpl]
						rmsNormInPlace(sl, m.PerLayerProjNorm, float32(cfg.RMSNormEps))
					}
					if m.EmbedPerLayer != nil && tokID < cfg.VocabPerLayer {
						embRow := m.EmbedPerLayer[tokID*totalDim : (tokID+1)*totalDim]
						for i := range proj {
							proj[i] = (proj[i] + embRow[i]*m.EmbedPerLayerScale) * m.PerLayerInputScale
						}
					}
					perLayerInputs = make([][]float32, nl)
					for ll := 0; ll < nl; ll++ {
						perLayerInputs[ll] = proj[ll*hpl : (ll+1)*hpl]
					}
				}
			}

			useDirectMLX := cfg.ModelType == "gemma4_text" || cfg.ModelType == "gemma3_text"
			for l := 0; l < len(g.Layers); l++ {
				layer := &g.Layers[l]
				cpuLayer := &m.Layers[l]

				// Per-layer dims
				layerHeadDim := headDim
				if cpuLayer.HeadDimLocal > 0 {
					layerHeadDim = cpuLayer.HeadDimLocal
				}
				qDim := numHeads * layerHeadDim
				layerKVDim := numKVHeads * layerHeadDim
				layerInter := inter
				if cpuLayer.GateWm != nil {
					layerInter = cpuLayer.GateWm.OutDim
				}

				// Save residual
				gpu.DevCopy(g.residual, g.hidden)

				// RMSNorm (GPU kernel with CPU fallback)
				gpu.DevRMSNorm(g.normed, g.hidden, layer.InputNorm, float32(cfg.RMSNormEps))
				if cfg.ModelType == "gemma4_text" {
					gpu.DevToBF16(g.normed, h)
				}

				if l == 0 && step == 0 {
					// gpu.Sync() — removed, all on GPU
				}
				// Q projection (always)
				if layer.QWmg != nil {
					if useDirectMLX {
						gpu.GemvMLXDirect(g.q, g.normed, layer.QWmg)
					} else {
						gpu.GemvMLX(g.q, g.normed, layer.QWmg)
					}
				} else if layer.QWg != nil {
					gpu.GemvQ4(g.q, g.normed, layer.QWg)
				} else if layer.QW != nil {
					g.gemv(g.q, g.normed, layer.QW, h, qDim)
				}

				// K/V projections (only for HasKV layers)
				if cpuLayer.HasKV {
					if layer.KWmg != nil {
						if useDirectMLX {
							gpu.GemvMLXDirect(g.k, g.normed, layer.KWmg)
						} else {
							gpu.GemvMLX(g.k, g.normed, layer.KWmg)
						}
						if useDirectMLX {
							gpu.GemvMLXDirect(g.v, g.normed, layer.VWmg)
						} else {
							gpu.GemvMLX(g.v, g.normed, layer.VWmg)
						}
					} else if layer.KWg != nil {
						gpu.GemvQ4(g.k, g.normed, layer.KWg)
						gpu.GemvQ4(g.v, g.normed, layer.VWg)
					} else if layer.KW != nil {
						g.gemv(g.k, g.normed, layer.KW, h, layerKVDim)
						g.gemv(g.v, g.normed, layer.VW, h, layerKVDim)
					}
				}

				if cfg.ModelType == "gemma4_text" {
					gpu.DevToBF16(g.q, qDim)
					if cpuLayer.HasKV {
						gpu.DevToBF16(g.k, layerKVDim)
						gpu.DevToBF16(g.v, layerKVDim)
					}
				}

				// Bias (Qwen2 only)
				if layer.QB != nil {
					gpu.DevAdd(g.q, g.q, layer.QB)
					if cpuLayer.HasKV {
						gpu.DevAdd(g.k, g.k, layer.KB)
						gpu.DevAdd(g.v, g.v, layer.VB)
					}
				}

				// V norm (Gemma4: RMSNormNoScale — no learned weight)
				if cfg.ModelType == "gemma4_text" && cpuLayer.HasKV {
					eps := float32(cfg.RMSNormEps)
					for head := 0; head < numKVHeads; head++ {
						vSlice := g.v.Slice(head*layerHeadDim, layerHeadDim)
						gpu.DevRMSNormNoScale(vSlice, vSlice, eps)
						gpu.DevToBF16(vSlice, layerHeadDim)
					}
				}

				// QK-Norm: RMSNorm each head
				if layer.QNorm != nil {
					for head := 0; head < numHeads; head++ {
						qSlice := g.q.Slice(head*layerHeadDim, layerHeadDim)
						gpu.DevRMSNorm(qSlice, qSlice, layer.QNorm, float32(cfg.RMSNormEps))
						if cfg.ModelType == "gemma4_text" {
							gpu.DevToBF16(qSlice, layerHeadDim)
						}
					}
					if cpuLayer.HasKV {
						for head := 0; head < numKVHeads; head++ {
							kSlice := g.k.Slice(head*layerHeadDim, layerHeadDim)
							gpu.DevRMSNorm(kSlice, kSlice, layer.KNorm, float32(cfg.RMSNormEps))
							if cfg.ModelType == "gemma4_text" {
								gpu.DevToBF16(kSlice, layerHeadDim)
							}
						}
					}
				}

				// RoPE: Gemma4 uses per-layer tables, others use global
				if cfg.ModelType == "gemma4_text" && g.ropeCosSinSWA != nil {
					isSWA := true
					if len(cfg.LayerTypes) > l {
						isSWA = cfg.LayerTypes[l] == "sliding_attention"
					}
					if isSWA {
						if !gpu.DevRoPEPartial(g.q, g.ropeCosSinSWA, pos, numHeads, layerHeadDim, m.RopeHalfSWA) {
							qd := g.q.Data()
							applyRoPEPartial(qd, m.RopeFreqsSWA, pos, numHeads, layerHeadDim, m.RopeHalfSWA)
							g.q.MarkDirty()
						}
						if cpuLayer.HasKV {
							if !gpu.DevRoPEPartial(g.k, g.ropeCosSinSWA, pos, numKVHeads, layerHeadDim, m.RopeHalfSWA) {
								kd3 := g.k.Data()
								applyRoPEPartial(kd3, m.RopeFreqsSWA, pos, numKVHeads, layerHeadDim, m.RopeHalfSWA)
								g.k.MarkDirty()
							}
						}
					} else {
						if !gpu.DevRoPEPartial(g.q, g.ropeCosSinFull, pos, numHeads, layerHeadDim, m.RopeHalfFull) {
							qd := g.q.Data()
							applyRoPEPartial(qd, m.RopeFreqsFull, pos, numHeads, layerHeadDim, m.RopeHalfFull)
							g.q.MarkDirty()
						}
						if cpuLayer.HasKV {
							if !gpu.DevRoPEPartial(g.k, g.ropeCosSinFull, pos, numKVHeads, layerHeadDim, m.RopeHalfFull) {
								kd3 := g.k.Data()
								applyRoPEPartial(kd3, m.RopeFreqsFull, pos, numKVHeads, layerHeadDim, m.RopeHalfFull)
								g.k.MarkDirty()
							}
						}
					}
				} else if g.ropeCosSin != nil && g.ropeCosSin.GPUPtr() != nil {
					gpu.DevRoPE(g.q, g.ropeCosSin, pos, numHeads, headDim)
					if cpuLayer.HasKV {
						gpu.DevRoPE(g.k, g.ropeCosSin, pos, numKVHeads, headDim)
					}
				} else {
					qd := g.q.Data()
					applyRoPE(qd, m.RopeFreqs, pos, numHeads, headDim)
					g.q.MarkDirty()
					if cpuLayer.HasKV {
						kd2 := g.k.Data()
						applyRoPE(kd2, m.RopeFreqs, pos, numKVHeads, headDim)
						g.k.MarkDirty()
					}
				}

				// KV cache: HasKV layers append, shared layers reuse source
				kvLayer := l
				if !cpuLayer.HasKV {
					kvLayer = cpuLayer.KVSourceLayer
				}
				seqLen := pos + 1

				if cpuLayer.HasKV && g.kvGPU_K[l] != nil && g.kvGPU_K[l].GPUPtr() != nil {
					g.k.ToGPU()
					g.v.ToGPU()
					kOff := gpu.CUdeviceptr(uint64(pos) * uint64(layerKVDim) * 4)
					gpu.CopyDtoD(g.kvGPU_K[l].GPUPtr().Ptr+kOff, g.k.GPUPtr().Ptr, uint64(layerKVDim*4))
					gpu.CopyDtoD(g.kvGPU_V[l].GPUPtr().Ptr+kOff, g.v.GPUPtr().Ptr, uint64(layerKVDim*4))
				} else if cpuLayer.HasKV {
					kd = g.k.Data()
					vd = g.v.Data()
					g.kvCacheK[l] = append(g.kvCacheK[l], kd[:layerKVDim]...)
					g.kvCacheV[l] = append(g.kvCacheV[l], vd[:layerKVDim]...)
				}

				// Attention (with per-layer headDim and scale)
				attnScale := float32(1.0 / math.Sqrt(float64(layerHeadDim)))
				if cfg.ModelType == "gemma4_text" {
					attnScale = 1.0
				}

				if g.kvGPU_K[kvLayer] != nil && g.kvGPU_K[kvLayer].GPUPtr() != nil {
					gpu.DevAttention(g.attnOut, g.q, g.kvGPU_K[kvLayer], g.kvGPU_V[kvLayer], seqLen, numHeads, numKVHeads, layerHeadDim, attnScale)
				} else {
					qd := g.q.Data()
					attnCPU := gqaAttentionScale(qd[:qDim], g.kvCacheK[kvLayer], g.kvCacheV[kvLayer], seqLen, numHeads, numKVHeads, layerHeadDim, attnScale)
					copy(g.attnOut.Data(), attnCPU)
					g.attnOut.MarkDirty()
				}

				// Output projection
				if layer.OWmg != nil {
					if useDirectMLX {
						gpu.GemvMLXDirect(g.oOut, g.attnOut, layer.OWmg)
					} else {
						gpu.GemvMLX(g.oOut, g.attnOut, layer.OWmg)
					}
				} else if layer.OWg != nil {
					g.attnOut.ToGPU()
					gpu.GemvQ4(g.oOut, g.attnOut, layer.OWg)
				} else if layer.OW != nil {
					g.gemv(g.oOut, g.attnOut, layer.OW, qDim, h)
				}

				// Gemma3/4: post-attn norm before residual, separate pre-FFN norm
				if layer.PreFFNNorm != nil {
					gpu.DevRMSNorm(g.oOut, g.oOut, layer.PostNorm, float32(cfg.RMSNormEps))
					gpu.DevAdd(g.hidden, g.residual, g.oOut)
					gpu.DevCopy(g.residual, g.hidden)
					gpu.DevRMSNorm(g.normed, g.hidden, layer.PreFFNNorm, float32(cfg.RMSNormEps))
				} else {
					gpu.DevAdd(g.hidden, g.residual, g.oOut)
					gpu.DevCopy(g.residual, g.hidden)
					gpu.DevRMSNorm(g.normed, g.hidden, layer.PostNorm, float32(cfg.RMSNormEps))
				}

				// MLP: gate + up projections
				if layer.GateWg != nil {
					g.normed.ToGPU()
					gpu.GemvQ4(g.gate, g.normed, layer.GateWg)
					gpu.GemvQ4(g.up, g.normed, layer.UpWg)
				} else if layer.GateWmg != nil {
					if useDirectMLX {
						gpu.GemvMLXDirect(g.gate, g.normed, layer.GateWmg)
					} else {
						gpu.GemvMLX(g.gate, g.normed, layer.GateWmg)
					}
					if useDirectMLX {
						gpu.GemvMLXDirect(g.up, g.normed, layer.UpWmg)
					} else {
						gpu.GemvMLX(g.up, g.normed, layer.UpWmg)
					}
				} else if layer.GateWq != nil {
					nd := g.normed.Data()
					gd := g.gate.Data()
					ud := g.up.Data()
					gemvQ4Sym(gd, nd, layer.GateWq.QWeight, layer.GateWq.GIdx, layer.GateWq.Scales, layer.GateWq.InDim, layer.GateWq.OutDim)
					gemvQ4Sym(ud, nd, layer.UpWq.QWeight, layer.UpWq.GIdx, layer.UpWq.Scales, layer.UpWq.InDim, layer.UpWq.OutDim)
					g.gate.MarkDirty()
					g.up.MarkDirty()
				} else {
					g.gemv(g.gate, g.normed, layer.GateW, h, inter)
					g.gemv(g.up, g.normed, layer.UpW, h, inter)
				}

				if cfg.ModelType == "gemma4_text" {
					gpu.DevToBF16(g.gate, layerInter)
					gpu.DevToBF16(g.up, layerInter)
				}

				// Activation(gate) * up
				if cfg.HiddenAct == "gelu_pytorch_tanh" {
					// GELU (Gemma3/4) — GPU kernel
					gpu.DevGELUTanhMul(g.gate, g.up, layerInter)
					if cfg.ModelType == "gemma4_text" {
						gpu.DevToBF16(g.gate, layerInter)
					}
				} else {
					gpu.DevSiLUMul(g.gate, g.gate, g.up)
				}

				// Down projection
				if layer.DownWmg != nil {
					if useDirectMLX {
						gpu.GemvMLXDirect(g.down, g.gate, layer.DownWmg)
					} else {
						gpu.GemvMLX(g.down, g.gate, layer.DownWmg)
					}
				} else if layer.DownWg != nil {
					g.gate.ToGPU()
					gpu.GemvQ4(g.down, g.gate, layer.DownWg)
				} else if layer.DownWmg != nil {
					if useDirectMLX {
						gpu.GemvMLXDirect(g.down, g.gate, layer.DownWmg)
					} else {
						gpu.GemvMLX(g.down, g.gate, layer.DownWmg)
					}
				} else if layer.DownWq != nil {
					gd := g.gate.Data()
					dd := g.down.Data()
					gemvQ4Sym(dd, gd, layer.DownWq.QWeight, layer.DownWq.GIdx, layer.DownWq.Scales, layer.DownWq.InDim, layer.DownWq.OutDim)
					g.down.MarkDirty()
				} else {
					g.gemv(g.down, g.gate, layer.DownW, inter, h)
				}

				if cfg.ModelType == "gemma4_text" {
					gpu.DevToBF16(g.down, h)
				}

				// Post-FFN norm (Gemma3/4)
				if layer.PostFFNNorm != nil {
					gpu.DevRMSNorm(g.down, g.down, layer.PostFFNNorm, float32(cfg.RMSNormEps))
					if cfg.ModelType == "gemma4_text" {
						gpu.DevToBF16(g.down, h)
					}
				}

				// Residual add
				gpu.DevAdd(g.hidden, g.residual, g.down)

				// Per-layer input gating (Gemma4, GPU path with CPU fallback)
				if layer.PLIGate != nil && usePLIGPU {
					hpl := cfg.HiddenPerLayer
					pliSlice := g.perLayerProjBuf.Slice(l*hpl, hpl)
					gpu.DevGemv(g.pliGateBuf, g.hidden, layer.PLIGate, hpl, h)
					gpu.DevGELUTanhMul(g.pliGateBuf, pliSlice, hpl)
					gpu.DevGemv(g.pliProjBuf, g.pliGateBuf, layer.PLIProj, h, hpl)
					gpu.DevRMSNorm(g.pliProjBuf, g.pliProjBuf, layer.PLIPostNorm, float32(cfg.RMSNormEps))
					gpu.DevAdd(g.hidden, g.hidden, g.pliProjBuf)
				} else if cpuLayer.PLIGate != nil && perLayerInputs != nil && l < len(perLayerInputs) {
					hpl := cfg.HiddenPerLayer
					pli := perLayerInputs[l]
					hd3 := g.hidden.Data()
					gate2 := make([]float32, hpl)
					gemvNT(gate2, hd3, cpuLayer.PLIGate, h, hpl)
					for i := range gate2 {
						gate2[i] = geluTanh(gate2[i])
					}
					for i := range gate2 {
						gate2[i] *= pli[i]
					}
					proj2 := make([]float32, h)
					gemvNT(proj2, gate2, cpuLayer.PLIProj, hpl, h)
					rmsNormInPlace(proj2, cpuLayer.PLIPostNorm, float32(cfg.RMSNormEps))
					for i := range hd3 {
						hd3[i] += proj2[i]
					}
					g.hidden.MarkDirty()
				}

				// Layer scalar (Gemma4)
				if cpuLayer.LayerScalar != 1.0 {
					hd3 := g.hidden.Data()
					for i := range hd3 {
						hd3[i] *= cpuLayer.LayerScalar
					}
					g.hidden.MarkDirty()
				}
				if cfg.ModelType == "gemma4_text" {
					gpu.DevToBF16(g.hidden, h)
				}
				if debugLayerHook != nil {
					debugLayerHook("gpu", step, l, g.hidden.Data())
				}
			}

		} // end !skipLayers

		// Sync GPU → CPU for final norm + sampling
		gpu.Sync() // drain all queued GPU work before readback

		if g.lmHeadGPU != nil {
			// GPU path: RMSNorm + GEMV on GPU, download logits
			gpu.DevRMSNorm(g.hidden, g.hidden, g.normGPU, float32(cfg.RMSNormEps))
			if cfg.ModelType == "gemma4_text" {
				gpu.DevToBF16(g.hidden, h)
			}
			// logits = lmHead[vocab,h] × hidden[h] → [vocab]
			gpu.DevLMHead(g.logitsGPU, g.hidden, g.lmHeadGPU, g.vocabSize, h)
			gpu.Sync()
			copy(logits, g.logitsGPU.Data())
		} else {
			hd = g.hidden.Data()
			rmsNormInPlace(hd, g.normWeight, float32(cfg.RMSNormEps))
			// Try chunked GPU LM head, fall back to parallel SIMD
			if !g.chunkedGPULMHead(logits, hd, g.vocabSize, h) {
				gemvNTParallel(logits, hd, g.lmHead, h, g.vocabSize)
			}
		}

		// Greedy sampling
		if step >= len(tokenIDs)-1 {
			bestID := 0
			bestVal := logits[0]
			for j := 1; j < g.vocabSize; j++ {
				if logits[j] > bestVal {
					bestVal = logits[j]
					bestID = j
				}
			}
			if debugLogitsHook != nil {
				debugLogitsHook("gpu", step, g.hidden.Data(), logits)
			}
			output = append(output, bestID)
		}
	}

	if len(output) > len(tokenIDs)+1 {
	}
	return output[len(tokenIDs):]
}
