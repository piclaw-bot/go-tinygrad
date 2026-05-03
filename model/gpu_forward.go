package model

// GPU-resident LLM forward pass using DevBuf.
// tinygrad approach: all weights + hidden state on GPU.
// Every op dispatches to GPU kernel with CPU fallback.

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/rcarmo/go-tinygrad/gpu"
	"github.com/rcarmo/go-tinygrad/tensor"
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
	lmHead     []float32 // [vocab, h]
	vocabSize  int
}

type gpuLayerBufs struct {
	QW, KW, VW, OW         *gpu.DevBuf
	QB, KB, VB             *gpu.DevBuf
	GateW, UpW, DownW      *gpu.DevBuf
	InputNorm, PostNorm    *gpu.DevBuf
	// Quantized path (stays CPU)
	QWq, KWq, VWq, OWq       *QuantWeight
	GateWq, UpWq, DownWq     *QuantWeight
	QWg, KWg, VWg, OWg       *gpu.GPUQuantWeight
	GateWg, UpWg, DownWg     *gpu.GPUQuantWeight
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

	// GPU KV cache mirrors
	g.kvGPU_K = make([]*gpu.DevBuf, len(m.Layers))

	g.kvGPU_V = make([]*gpu.DevBuf, len(m.Layers))

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

	for step := 0; step < len(tokenIDs)+maxTokens-1; step++ {
		var tokID int
		if step < len(tokenIDs) {
			tokID = tokenIDs[step]
		} else {
			tokID = output[len(output)-1]
		}
		pos := step

		// Embedding (CPU — vocab too large for VRAM on small GPUs)
		embData := m.EmbedTokens.Data()
		hd := g.hidden.Data()
		copy(hd, embData[tokID*h:(tokID+1)*h])
		g.hidden.MarkDirty() // CPU data set
		if step == 0 {
		}

		for l := 0; l < len(g.Layers); l++ {
			if l > 0 && true { gpu.Sync() } // flush GPU queue periodically
			layer := &g.Layers[l]

			// Save residual
			gpu.DevCopy(g.residual, g.hidden)

			// RMSNorm (GPU kernel with CPU fallback)
			gpu.DevRMSNorm(g.normed, g.hidden, layer.InputNorm, float32(cfg.RMSNormEps))

			if l == 0 && step == 0 {
				gpu.Sync()
			}
			// Q/K/V projections
			if layer.QWg != nil {
				gpu.GemvQ4(g.q, g.normed, layer.QWg)
				gpu.GemvQ4(g.k, g.normed, layer.KWg)
				gpu.GemvQ4(g.v, g.normed, layer.VWg)
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
				gpu.Sync()
			}
			// Bias (GPU add or CPU)
			
			if layer.QB != nil {
				gpu.DevAdd(g.q, g.q, layer.QB)
				gpu.DevAdd(g.k, g.k, layer.KB)
				gpu.DevAdd(g.v, g.v, layer.VB)
			}

			// RoPE (GPU with precomputed cos/sin, or CPU fallback)
			gpu.Sync()
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
			if g.kvGPU_K[l] != nil && g.kvGPU_K[l].GPUPtr() != nil && g.k.GPUPtr() != nil {
				// GPU path: CopyDtoD K/V into cache at position offset
				kOff := gpu.CUdeviceptr(uint64(pos) * uint64(kvDim) * 4)
				gpu.CopyDtoD(g.kvGPU_K[l].GPUPtr().Ptr+kOff, g.k.GPUPtr().Ptr, uint64(kvDim*4))
				gpu.CopyDtoD(g.kvGPU_V[l].GPUPtr().Ptr+kOff, g.v.GPUPtr().Ptr, uint64(kvDim*4))
				// GPU attention
				gpu.DevAttention(g.attnOut, g.q, g.kvGPU_K[l], g.kvGPU_V[l], seqLen, numHeads, numKVHeads, headDim)
			} else {
				// CPU fallback
				gpu.Sync()
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
				gpu.GemvQ4(g.oOut, g.attnOut, layer.OWg)
			} else if layer.OWq != nil {
				oOut := g.oOut.Data()
				gemvQ4Sym(oOut, g.attnOut.Data(), layer.OWq.QWeight, layer.OWq.GIdx, layer.OWq.Scales, layer.OWq.InDim, layer.OWq.OutDim)
				g.oOut.MarkDirty()
			} else {
				g.gemv(g.oOut, g.attnOut, layer.OW, h, h)
			}

			// Residual add (GPU)
			
			gpu.DevAdd(g.hidden, g.residual, g.oOut)

			// Post-attention norm
			gpu.DevCopy(g.residual, g.hidden)
			gpu.DevRMSNorm(g.normed, g.hidden, layer.PostNorm, float32(cfg.RMSNormEps))

			// MLP: gate + up projections
				nd := g.normed.Data()
			if layer.GateWq != nil {
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
			gpu.DevSiLU(g.gate, g.gate)
			gpu.DevMul(g.gate, g.gate, g.up)

			// Down projection
			if layer.DownWg != nil {
				gpu.GemvQ4(g.down, g.gate, layer.DownWg)
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
				gpu.Sync()
			}
		}

		// Sync GPU → CPU for final norm + sampling
		gpu.Sync()
		hd = g.hidden.Data()
		rmsNormInPlace(hd, g.normWeight, float32(cfg.RMSNormEps))

		// LM head (CPU — vocab × h matmul)
		for j := 0; j < g.vocabSize; j++ {
			sum := float32(0)
			row := g.lmHead[j*h : (j+1)*h]
			for p := 0; p < h; p++ {
				sum += hd[p] * row[p]
			}
			logits[j] = sum
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

