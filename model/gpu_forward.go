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

	// KV cache (GPU-resident for fast path, CPU for fallback)
	kvCacheK, kvCacheV [][]float32 // CPU slices
	kvGPU_K, kvGPU_V   []*gpu.DevBuf // GPU buffers [maxSeq * kvDim] per layer

	// RoPE precomputed cos/sin
	ropeCosSin *gpu.DevBuf

	// Final norm + lm_head stay on CPU (vocab is huge)
	normWeight []float32

	// GPU LM head
	lmHeadGPU *gpu.DevBuf // [vocab × h] F32 on GPU
	normGPU   *gpu.DevBuf // final norm weights on GPU
	logitsGPU *gpu.DevBuf // [vocab] logits output on GPU
	lmHead     []float32 // [vocab, h]
	vocabSize  int
}

type gpuLayerBufs struct {
	QW, KW, VW, OW         *gpu.DevBuf
	QB, KB, VB             *gpu.DevBuf
	QNorm, KNorm           *gpu.DevBuf // QK-Norm (Qwen3)
	GateW, UpW, DownW      *gpu.DevBuf
	InputNorm, PostNorm    *gpu.DevBuf
	// GPTQ quantized (CPU fallback)
	QWq, KWq, VWq, OWq       *QuantWeight
	GateWq, UpWq, DownWq     *QuantWeight
	// GPTQ on GPU
	QWg, KWg, VWg, OWg       *gpu.GPUQuantWeight
	GateWg, UpWg, DownWg     *gpu.GPUQuantWeight
	// MLX on GPU
	QWmg, KWmg, VWmg, OWmg       *gpu.GPUMLXWeight
	GateWmg, UpWmg, DownWmg     *gpu.GPUMLXWeight
	// MLX on CPU
	QWm, KWm, VWm, OWm           *MLXQuantWeight
	GateWm, UpWm, DownWm         *MLXQuantWeight
}

// LoadGPUModel uploads model weights to GPU using DevBuf.
func LoadGPUModel(m *LlamaModel) (*GPUModel, error) {
	runtime.LockOSThread()
	start := time.Now()

	cfg := m.Config
	h := cfg.HiddenSize
	kvDim := h * cfg.NumKVHeads / cfg.NumHeads
	inter := cfg.Intermediate

	g := &GPUModel{
		CPU:       m,
		Config:    cfg,
		vocabSize: cfg.VocabSize,
	}

	// Work buffers
	g.hidden = gpu.NewDevBuf(h)
	g.residual = gpu.NewDevBuf(h)
	g.normed = gpu.NewDevBuf(h)
	g.q = gpu.NewDevBuf(h)
	g.k = gpu.NewDevBuf(kvDim)
	g.v = gpu.NewDevBuf(kvDim)
	g.attnOut = gpu.NewDevBuf(h)
	g.oOut = gpu.NewDevBuf(h)
	g.gate = gpu.NewDevBuf(inter)
	g.up = gpu.NewDevBuf(inter)
	g.down = gpu.NewDevBuf(h)

	// Try to move work buffers to GPU
	useGPU := true
	for _, buf := range []*gpu.DevBuf{g.hidden, g.residual, g.normed, g.q, g.k, g.v, g.attnOut, g.oOut, g.gate, g.up, g.down} {
		if err := buf.ToGPU(); err != nil {
			useGPU = false
			break
		}
	}

	// Upload per-layer weights
	wrapTensor := func(t *tensor.Tensor) *gpu.DevBuf {
		if t == nil { return nil }
		b := gpu.NewDevBufFrom(t.Data())
		if useGPU { b.ToGPU() }
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
			gl.QWq = layer.QWq; gl.KWq = layer.KWq; gl.VWq = layer.VWq; gl.OWq = layer.OWq
			gl.GateWq = layer.GateWq; gl.UpWq = layer.UpWq; gl.DownWq = layer.DownWq
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
			gl.QWm = layer.QWm; gl.KWm = layer.KWm; gl.VWm = layer.VWm; gl.OWm = layer.OWm
			gl.GateWm = layer.GateWm; gl.UpWm = layer.UpWm; gl.DownWm = layer.DownWm
			if gpu.SgemmReady() {
				um := func(qw *MLXQuantWeight) *gpu.GPUMLXWeight {
					w, err := gpu.UploadMLXWeight(qw.Weight, qw.Scales, qw.Biases, qw.InDim, qw.OutDim, qw.GroupSize)
					if err != nil && i == 0 { fmt.Printf("[gpu] MLX upload %dx%d: %v\n", qw.OutDim, qw.InDim, err) }
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

		gl.QB = wrapTensor(layer.QB)
		gl.KB = wrapTensor(layer.KB)
		gl.VB = wrapTensor(layer.VB)
		gl.QNorm = wrapTensor(layer.QNorm)
		gl.KNorm = wrapTensor(layer.KNorm)

		g.Layers[i] = gl
	}

	// KV cache
	g.kvCacheK = make([][]float32, len(m.Layers))
	g.kvCacheV = make([][]float32, len(m.Layers))
	for l := range g.kvCacheK {
		g.kvCacheK[l] = make([]float32, 0, 2048*kvDim)
		g.kvCacheV[l] = make([]float32, 0, 2048*kvDim)
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

	device := "CPU"
	// Precompute RoPE cos/sin table
	{
		headDimL := h / cfg.NumHeads
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

	// GPU KV cache: pre-allocate for all layers
	maxSeq := 2048
	g.kvGPU_K = make([]*gpu.DevBuf, len(m.Layers))
	g.kvGPU_V = make([]*gpu.DevBuf, len(m.Layers))
	for i := range g.kvGPU_K {
		g.kvGPU_K[i] = gpu.NewDevBuf(maxSeq * kvDim)
		g.kvGPU_V[i] = gpu.NewDevBuf(maxSeq * kvDim)
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
	if useGPU { device = "GPU" }
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
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := h / numHeads
	kvDim := headDim * numKVHeads
	inter := cfg.Intermediate
	m := g.CPU

	output := make([]int, len(tokenIDs), len(tokenIDs)+maxTokens)
	copy(output, tokenIDs)

	// Temp CPU buffers for RoPE + attention (sequential ops)
	qCPU := make([]float32, h)
	var kd, vd []float32
	kCPU := make([]float32, kvDim)
	vCPU := make([]float32, kvDim)
	logits := make([]float32, g.vocabSize)

	// Batched prefill: process all prompt tokens at once
	prefillStart := 0
	if len(tokenIDs) > 1 && gpu.BatchGEMMReady() {
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
		g.hidden.MarkDirty() // CPU data set
		if step == 0 {
		}

		
		for l := 0; l < len(g.Layers); l++ {
			// Debug: trace each GPU op
						layer := &g.Layers[l]

			// Save residual

			// RMSNorm (GPU kernel with CPU fallback)

			if l == 0 && step == 0 {
			// gpu.Sync() — removed, all on GPU
			}
			// Q/K/V projections
			if layer.QWg != nil {
				// GPTQ GPU path
				gpu.PrefetchWeights(layer.OWg, layer.GateWg, layer.UpWg, layer.DownWg)
				gpu.GemvQ4(g.q, g.normed, layer.QWg)
				gpu.GemvQ4(g.k, g.normed, layer.KWg)
				gpu.GemvQ4(g.v, g.normed, layer.VWg)
				gpu.WaitPrefetch()
			} else if layer.QWmg != nil {
				// MLX GPU path
				gpu.GemvMLX(g.q, g.normed, layer.QWmg)
				gpu.GemvMLX(g.k, g.normed, layer.KWmg)
				gpu.GemvMLX(g.v, g.normed, layer.VWmg)
			} else if layer.QWq != nil {
				nd := g.normed.Data()
				gemvQ4Sym(qCPU, nd, layer.QWq.QWeight, layer.QWq.GIdx, layer.QWq.Scales, layer.QWq.InDim, layer.QWq.OutDim)
				gemvQ4Sym(kCPU, nd, layer.KWq.QWeight, layer.KWq.GIdx, layer.KWq.Scales, layer.KWq.InDim, layer.KWq.OutDim)
				gemvQ4Sym(vCPU, nd, layer.VWq.QWeight, layer.VWq.GIdx, layer.VWq.Scales, layer.VWq.InDim, layer.VWq.OutDim)
				copy(g.q.Data(), qCPU)
				copy(g.k.Data(), kCPU)
				copy(g.v.Data(), vCPU)
				g.q.MarkDirty(); g.k.MarkDirty(); g.v.MarkDirty()
			} else {
				g.gemv(g.q, g.normed, layer.QW, h, h)
				g.gemv(g.k, g.normed, layer.KW, h, kvDim)
				g.gemv(g.v, g.normed, layer.VW, h, kvDim)
			}

			if l == 0 && step == 0 {
			// gpu.Sync() — removed, all on GPU
			}
			// Bias (GPU add or CPU)
			
			if layer.QB != nil {
				gpu.DevAdd(g.q, g.q, layer.QB)
				gpu.DevAdd(g.k, g.k, layer.KB)
				gpu.DevAdd(g.v, g.v, layer.VB)
			}

			// QK-Norm (Qwen3): RMSNorm each head independently
			if layer.QNorm != nil {
				// Per-head RMSNorm on Q and K
				for head := 0; head < numHeads; head++ {
					qSlice := g.q.Slice(head*headDim, headDim)
					gpu.DevRMSNorm(qSlice, qSlice, layer.QNorm, float32(cfg.RMSNormEps))
				}
				for head := 0; head < numKVHeads; head++ {
					kSlice := g.k.Slice(head*headDim, headDim)
					gpu.DevRMSNorm(kSlice, kSlice, layer.KNorm, float32(cfg.RMSNormEps))
				}
			}

			// RoPE (GPU with precomputed cos/sin, or CPU fallback)
			// gpu.Sync() — removed, all on GPU
			if g.ropeCosSin != nil && g.ropeCosSin.GPUPtr() != nil {
				gpu.DevRoPE(g.q, g.ropeCosSin, pos, numHeads, headDim)
				gpu.DevRoPE(g.k, g.ropeCosSin, pos, numKVHeads, headDim)
			} else {
				qd := g.q.Data()
				kd2 := g.k.Data()
				applyRoPE(qd, m.RopeFreqs, pos, numHeads, headDim)
				applyRoPE(kd2, m.RopeFreqs, pos, numKVHeads, headDim)
				g.q.MarkDirty()
				g.k.MarkDirty()
			}

			// KV cache append (GPU-resident, no host download)
			seqLen := pos + 1
			if g.kvGPU_K[l] != nil && g.kvGPU_K[l].GPUPtr() != nil {
				g.k.ToGPU()
				g.v.ToGPU()
				// GPU path: CopyDtoD K/V into cache at position offset
				kOff := gpu.CUdeviceptr(uint64(pos) * uint64(kvDim) * 4)
				gpu.CopyDtoD(g.kvGPU_K[l].GPUPtr().Ptr+kOff, g.k.GPUPtr().Ptr, uint64(kvDim*4))
				gpu.CopyDtoD(g.kvGPU_V[l].GPUPtr().Ptr+kOff, g.v.GPUPtr().Ptr, uint64(kvDim*4))
				// GPU attention
				gpu.DevAttention(g.attnOut, g.q, g.kvGPU_K[l], g.kvGPU_V[l], seqLen, numHeads, numKVHeads, headDim)
			} else {
				// CPU fallback
			// gpu.Sync() — removed, all on GPU
				kd = g.k.Data()
				vd = g.v.Data()
				g.kvCacheK[l] = append(g.kvCacheK[l], kd...)
				g.kvCacheV[l] = append(g.kvCacheV[l], vd...)
				qd := g.q.Data()
				attnCPU := gqaAttention(qd, g.kvCacheK[l], g.kvCacheV[l], seqLen, numHeads, numKVHeads, headDim)
				copy(g.attnOut.Data(), attnCPU)
				g.attnOut.MarkDirty()
			}

			// Output projection (GPU GEMV)
			
			if layer.OWg != nil {
				g.attnOut.ToGPU()
				gpu.GemvQ4(g.oOut, g.attnOut, layer.OWg)
			} else if layer.OWmg != nil {
				gpu.GemvMLX(g.oOut, g.attnOut, layer.OWmg)
			} else if layer.OWq != nil {
				oOut := g.oOut.Data()
				gemvQ4Sym(oOut, g.attnOut.Data(), layer.OWq.QWeight, layer.OWq.GIdx, layer.OWq.Scales, layer.OWq.InDim, layer.OWq.OutDim)
				g.oOut.MarkDirty()
			} else {
				g.gemv(g.oOut, g.attnOut, layer.OW, h, h)
			}

			// Residual add (GPU)
			gpu.DevAdd(g.hidden, g.residual, g.oOut)
			// Prefetch next layer's attention weights
			if l+1 < len(g.Layers) {
				next := &g.Layers[l+1]
				gpu.PrefetchWeights(next.QWg, next.KWg, next.VWg)
			}

			// Post-attention norm
			gpu.DevRMSNorm(g.normed, g.hidden, layer.PostNorm, float32(cfg.RMSNormEps))

			// MLP: gate + up projections
			if layer.GateWg != nil {
				g.normed.ToGPU()
				gpu.GemvQ4(g.gate, g.normed, layer.GateWg)
				gpu.GemvQ4(g.up, g.normed, layer.UpWg)
			} else if layer.GateWmg != nil {
				gpu.GemvMLX(g.gate, g.normed, layer.GateWmg)
				gpu.GemvMLX(g.up, g.normed, layer.UpWmg)
			} else if layer.GateWq != nil {
				nd := g.normed.Data()
				gd := g.gate.Data()
				ud := g.up.Data()
				gemvQ4Sym(gd, nd, layer.GateWq.QWeight, layer.GateWq.GIdx, layer.GateWq.Scales, layer.GateWq.InDim, layer.GateWq.OutDim)
				gemvQ4Sym(ud, nd, layer.UpWq.QWeight, layer.UpWq.GIdx, layer.UpWq.Scales, layer.UpWq.InDim, layer.UpWq.OutDim)
				g.gate.MarkDirty(); g.up.MarkDirty()
			} else {
				g.gemv(g.gate, g.normed, layer.GateW, h, inter)
				g.gemv(g.up, g.normed, layer.UpW, h, inter)
			}

			// SiLU(gate) * up (GPU)
			gpu.DevSiLUMul(g.gate, g.gate, g.up) // fused: 2 kernels -> 1

			// Down projection
			if layer.DownWg != nil {
				g.gate.ToGPU()
				gpu.GemvQ4(g.down, g.gate, layer.DownWg)
			} else if layer.DownWmg != nil {
				gpu.GemvMLX(g.down, g.gate, layer.DownWmg)
			} else if layer.DownWq != nil {
				gd := g.gate.Data()
				dd := g.down.Data()
				gemvQ4Sym(dd, gd, layer.DownWq.QWeight, layer.DownWq.GIdx, layer.DownWq.Scales, layer.DownWq.InDim, layer.DownWq.OutDim)
				g.down.MarkDirty()
			} else {
				g.gemv(g.down, g.gate, layer.DownW, inter, h)
			}

			// Residual add
			
			gpu.DevAdd(g.hidden, g.residual, g.down)

			if l == 0 && step == 0 {
			// gpu.Sync() — removed, all on GPU
			}
		}

		} // end !skipLayers

		// Sync GPU → CPU for final norm + sampling
		gpu.Sync() // drain all queued GPU work before readback

		if g.lmHeadGPU != nil {
			// GPU path: RMSNorm + GEMV on GPU, download logits
			gpu.DevRMSNorm(g.hidden, g.hidden, g.normGPU, float32(cfg.RMSNormEps))
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
			output = append(output, bestID)
		}
	}

	if len(output) > len(tokenIDs)+1 {
	}
	return output[len(tokenIDs):]
}

