package model

// GPU-resident LLM forward pass using DevBuf.
// tinygrad approach: ALL computation on GPU. Only download logits for sampling.

import (
	"fmt"
	"math"
	"time"

	"github.com/rcarmo/go-tinygrad/gpu"
	"github.com/rcarmo/go-tinygrad/tensor"
)

// GPUModel wraps a LlamaModel with GPU-resident weights and buffers.
type GPUModel struct {
	CPU    *LlamaModel
	Config LlamaConfig

	Layers []gpuLayerBufs

	// Work buffers
	hidden, residual, normed *gpu.DevBuf
	q, k, v, attnOut, oOut   *gpu.DevBuf
	gate, up, down           *gpu.DevBuf

	// KV cache on GPU
	kvK, kvV []*gpu.DevBuf // [layer] each [maxSeq * kvDim]
	kvPos    int           // current position in cache

	// RoPE cos/sin table on GPU: [maxSeq, headDim] interleaved [cos0,sin0,cos1,sin1,...]
	ropeCosSin *gpu.DevBuf

	// Final layers
	normWeight *gpu.DevBuf
	lmHeadGPU  *gpu.DevBuf
	logitsBuf  *gpu.DevBuf
	vocabSize  int
}

type gpuLayerBufs struct {
	QW, KW, VW, OW         *gpu.DevBuf
	QB, KB, VB             *gpu.DevBuf
	GateW, UpW, DownW      *gpu.DevBuf
	InputNorm, PostNorm    *gpu.DevBuf
	QWq, KWq, VWq, OWq       *QuantWeight
	GateWq, UpWq, DownWq     *QuantWeight
	QWg, KWg, VWg, OWg       *gpu.GPUQuantWeight
	GateWg, UpWg, DownWg     *gpu.GPUQuantWeight
}

// LoadGPUModel uploads model weights to GPU using DevBuf.
func LoadGPUModel(m *LlamaModel) (*GPUModel, error) {
	start := time.Now()

	cfg := m.Config
	h := cfg.HiddenSize
	kvDim := h * cfg.NumKVHeads / cfg.NumHeads
	inter := cfg.Intermediate
	maxSeq := 2048

	g := &GPUModel{CPU: m, Config: cfg, vocabSize: cfg.VocabSize}

	// Work buffers → GPU
	g.hidden = gpu.NewDevBuf(h); g.hidden.ToGPU()
	g.residual = gpu.NewDevBuf(h); g.residual.ToGPU()
	g.normed = gpu.NewDevBuf(h); g.normed.ToGPU()
	g.q = gpu.NewDevBuf(h); g.q.ToGPU()
	g.k = gpu.NewDevBuf(kvDim); g.k.ToGPU()
	g.v = gpu.NewDevBuf(kvDim); g.v.ToGPU()
	g.attnOut = gpu.NewDevBuf(h); g.attnOut.ToGPU()
	g.oOut = gpu.NewDevBuf(h); g.oOut.ToGPU()
	g.gate = gpu.NewDevBuf(inter); g.gate.ToGPU()
	g.up = gpu.NewDevBuf(inter); g.up.ToGPU()
	g.down = gpu.NewDevBuf(h); g.down.ToGPU()

	// Precompute RoPE cos/sin table (CPU precision, uploaded to GPU)
	{
		headDimL := h / cfg.NumHeads
		halfDim := headDimL / 2
		maxSeqL := 2048
		cosSinData := make([]float32, maxSeqL*headDimL)
		for pos := 0; pos < maxSeqL; pos++ {
			for i := 0; i < halfDim; i++ {
				freq := m.RopeFreqs[pos*halfDim+i]
				cosSinData[pos*headDimL+i*2] = float32(math.Cos(float64(freq)))
				cosSinData[pos*headDimL+i*2+1] = float32(math.Sin(float64(freq)))
			}
		}
		g.ropeCosSin = gpu.NewDevBufFrom(cosSinData)
		g.ropeCosSin.ToGPU()
	}

	// KV cache → GPU
	g.kvK = make([]*gpu.DevBuf, len(m.Layers))
	g.kvV = make([]*gpu.DevBuf, len(m.Layers))
	for l := range m.Layers {
		g.kvK[l] = gpu.NewDevBuf(maxSeq * kvDim); g.kvK[l].ToGPU()
		g.kvV[l] = gpu.NewDevBuf(maxSeq * kvDim); g.kvV[l].ToGPU()
	}

	// Upload weights
	wrapTensor := func(t *tensor.Tensor) *gpu.DevBuf {
		if t == nil { return nil }
		b := gpu.NewDevBufFrom(t.Data())
		b.ToGPU()
		return b
	}

	g.Layers = make([]gpuLayerBufs, len(m.Layers))
	for i, layer := range m.Layers {
		gl := gpuLayerBufs{
			InputNorm: wrapTensor(layer.InputNorm),
			PostNorm:  wrapTensor(layer.PostNorm),
			QB: wrapTensor(layer.QB),
			KB: wrapTensor(layer.KB),
			VB: wrapTensor(layer.VB),
		}
		if layer.QWq != nil {
			gl.QWq = layer.QWq; gl.KWq = layer.KWq; gl.VWq = layer.VWq; gl.OWq = layer.OWq
			gl.GateWq = layer.GateWq; gl.UpWq = layer.UpWq; gl.DownWq = layer.DownWq
			if gpu.Q4Ready() {
				gl.QWg, _ = gpu.UploadQuantWeight(layer.QWq.QWeight, layer.QWq.GIdx, layer.QWq.Scales, layer.QWq.InDim, layer.QWq.OutDim)
				gl.KWg, _ = gpu.UploadQuantWeight(layer.KWq.QWeight, layer.KWq.GIdx, layer.KWq.Scales, layer.KWq.InDim, layer.KWq.OutDim)
				gl.VWg, _ = gpu.UploadQuantWeight(layer.VWq.QWeight, layer.VWq.GIdx, layer.VWq.Scales, layer.VWq.InDim, layer.VWq.OutDim)
				gl.OWg, _ = gpu.UploadQuantWeight(layer.OWq.QWeight, layer.OWq.GIdx, layer.OWq.Scales, layer.OWq.InDim, layer.OWq.OutDim)
				gl.GateWg, _ = gpu.UploadQuantWeight(layer.GateWq.QWeight, layer.GateWq.GIdx, layer.GateWq.Scales, layer.GateWq.InDim, layer.GateWq.OutDim)
				gl.UpWg, _ = gpu.UploadQuantWeight(layer.UpWq.QWeight, layer.UpWq.GIdx, layer.UpWq.Scales, layer.UpWq.InDim, layer.UpWq.OutDim)
				gl.DownWg, _ = gpu.UploadQuantWeight(layer.DownWq.QWeight, layer.DownWq.GIdx, layer.DownWq.Scales, layer.DownWq.InDim, layer.DownWq.OutDim)
			}
		} else {
			gl.QW = wrapTensor(layer.QW); gl.KW = wrapTensor(layer.KW)
			gl.VW = wrapTensor(layer.VW); gl.OW = wrapTensor(layer.OW)
			gl.GateW = wrapTensor(layer.GateW); gl.UpW = wrapTensor(layer.UpW)
			gl.DownW = wrapTensor(layer.DownW)
		}
		g.Layers[i] = gl
	}

	// Final norm + LM head → GPU
	g.normWeight = gpu.NewDevBufFrom(m.Norm.Data()); g.normWeight.ToGPU()
	lmData := m.EmbedTokens.Data()
	if m.LMHead != nil { lmData = m.LMHead.Data() }
	g.lmHeadGPU = gpu.NewDevBufFrom(lmData); g.lmHeadGPU.ToGPU()
	g.logitsBuf = gpu.NewDevBuf(cfg.VocabSize); g.logitsBuf.ToGPU()

	elapsed := time.Since(start)
	fmt.Printf("[model] Weights on GPU (%d layers, %v)\n", len(g.Layers), elapsed.Round(time.Millisecond))
	return g, nil
}

// gemv dispatches to GPU Q4, GPU F32, or CPU based on what's available.
func (g *GPUModel) gemv(out, x *gpu.DevBuf, layer *gpuLayerBufs, which string) {
	var qw *gpu.GPUQuantWeight
	var fw *gpu.DevBuf
	switch which {
	case "q": qw, fw = layer.QWg, layer.QW
	case "k": qw, fw = layer.KWg, layer.KW
	case "v": qw, fw = layer.VWg, layer.VW
	case "o": qw, fw = layer.OWg, layer.OW
	case "gate": qw, fw = layer.GateWg, layer.GateW
	case "up": qw, fw = layer.UpWg, layer.UpW
	case "down": qw, fw = layer.DownWg, layer.DownW
	}
	if qw != nil {
		gpu.GemvQ4(out, x, qw)
		return
	}
	if fw != nil {
		h := g.Config.HiddenSize
		if g.CPU.Large {
			outDim := out.Len()
			gpu.DevGemv(out, x, fw, outDim, h)
		} else {
			gpu.DevGemvNN(out, x, fw, h, out.Len())
		}
		return
	}
}

// Generate produces tokens with fully GPU-resident forward pass.
func (g *GPUModel) Generate(tokenIDs []int, maxTokens int) []int {
	cfg := g.Config
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := h / numHeads
	kvDim := headDim * numKVHeads

	output := make([]int, len(tokenIDs), len(tokenIDs)+maxTokens)
	copy(output, tokenIDs)

	logitsCPU := make([]float32, g.vocabSize)
	g.kvPos = 0

	for step := 0; step < len(tokenIDs)+maxTokens-1; step++ {
		var tokID int
		if step < len(tokenIDs) {
			tokID = tokenIDs[step]
		} else {
			tokID = output[len(output)-1]
		}
		pos := step

		// Embedding → GPU (only CPU→GPU transfer per token)
		embData := g.CPU.EmbedTokens.Data()
		copy(g.hidden.Data(), embData[tokID*h:(tokID+1)*h])
		g.hidden.MarkDirty()

		// === ALL GPU from here — no sync until logits ===

		for l := 0; l < len(g.Layers); l++ {
			layer := &g.Layers[l]

			// Residual save (GPU memcpy)
			gpu.DevCopy(g.residual, g.hidden)

			// RMSNorm (GPU kernel)
			gpu.DevRMSNorm(g.normed, g.hidden, layer.InputNorm, float32(cfg.RMSNormEps))

			// Q/K/V projections (GPU Q4 or F32 GEMV)
			g.gemv(g.q, g.normed, layer, "q")
			g.gemv(g.k, g.normed, layer, "k")
			g.gemv(g.v, g.normed, layer, "v")

			// Bias add (GPU)
			if layer.QB != nil {
				gpu.DevAdd(g.q, g.q, layer.QB)
				gpu.DevAdd(g.k, g.k, layer.KB)
				gpu.DevAdd(g.v, g.v, layer.VB)
			}

			// RoPE on Q and K (GPU kernel with CPU-precision cos/sin)
			gpu.DevRoPE(g.q, g.ropeCosSin, pos, numHeads, headDim)
			gpu.DevRoPE(g.k, g.ropeCosSin, pos, numKVHeads, headDim)

			// Append K,V to GPU KV cache
			// Copy k[kvDim] to kvK[l][pos*kvDim : (pos+1)*kvDim]
			// Use cuMemcpyDtoD for GPU-to-GPU copy within the cache
			if g.k.GPUPtr() != nil && g.kvK[l].GPUPtr() != nil {
				offset := gpu.CUdeviceptr(uint64(pos) * uint64(kvDim) * 4)
				gpu.CopyDtoD(g.kvK[l].GPUPtr().Ptr+offset, g.k.GPUPtr().Ptr, uint64(kvDim*4))
				gpu.CopyDtoD(g.kvV[l].GPUPtr().Ptr+offset, g.v.GPUPtr().Ptr, uint64(kvDim*4))
			}

			// Attention (GPU kernel — one block per head)
			seqLen := pos + 1
			gpu.DevAttention(g.attnOut, g.q, g.kvK[l], g.kvV[l], seqLen, numHeads, numKVHeads, headDim)

			// O projection (GPU)
			g.gemv(g.oOut, g.attnOut, layer, "o")

			// Residual add (GPU)
			gpu.DevAdd(g.hidden, g.residual, g.oOut)

			// Post-attention norm + MLP
			gpu.DevCopy(g.residual, g.hidden)
			gpu.DevRMSNorm(g.normed, g.hidden, layer.PostNorm, float32(cfg.RMSNormEps))

			g.gemv(g.gate, g.normed, layer, "gate")
			g.gemv(g.up, g.normed, layer, "up")

			gpu.DevSiLU(g.gate, g.gate)
			gpu.DevMul(g.gate, g.gate, g.up)

			g.gemv(g.down, g.gate, layer, "down")

			gpu.DevAdd(g.hidden, g.residual, g.down)
		}

		// Final norm (GPU)
		gpu.DevRMSNorm(g.hidden, g.hidden, g.normWeight, float32(cfg.RMSNormEps))

		// LM head (GPU GEMV: logits = hidden @ lmHead^T)
		gpu.DevGemv(g.logitsBuf, g.hidden, g.lmHeadGPU, g.vocabSize, h)

		// === ONLY sync + download here ===
		gpu.Sync()
		copy(logitsCPU, g.logitsBuf.Data())

		// Greedy sampling (CPU)
		if step >= len(tokenIDs)-1 {
			bestID := 0
			bestVal := logitsCPU[0]
			for j := 1; j < g.vocabSize; j++ {
				if logitsCPU[j] > bestVal {
					bestVal = logitsCPU[j]
					bestID = j
				}
			}
			output = append(output, bestID)
		}
	}
	return output[len(tokenIDs):]
}

// CopyDtoD wraps cuMemcpyDtoD for direct GPU→GPU copy.
// Exported from gpu package.
func init() {
	// DevAttention fallback uses CPU — handled in gpu package
}

// applyRoPE is defined in llama.go (CPU path)
// gqaAttention is defined in llama.go (CPU path)
// rmsNormInPlace is defined in llama.go (CPU path)

// For CPU fallback when GPU not available, Generate falls back to CPU model
func (g *GPUModel) generateCPUFallback(tokenIDs []int, maxTokens int) []int {
	return g.CPU.Generate(tokenIDs, maxTokens)
}

// Math helpers
var _ = math.Sqrt // keep import
