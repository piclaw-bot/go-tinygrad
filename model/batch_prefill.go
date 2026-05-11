package model

// Batched prefill: process all prompt tokens through the model at once.
// Reads each weight matrix once instead of B times → B× faster prefill.
//
// For a 6-token prompt with 28 layers:
//   Sequential: 28 × 6 × 7 GEMV = 1176 weight reads
//   Batched:    28 × 7 GEMM     = 196 weight reads (6× fewer)

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/gpu"
)

// prefillGPU processes all prompt tokens through the model in one batched pass.
// Returns the hidden state for the last token, with KV cache filled for all positions.
func (g *GPUModel) prefillGPU(tokenIDs []int) []float32 {
	cfg := g.CPU.Config
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := h / numHeads
	defaultScale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvDim := headDim * numKVHeads
	B := len(tokenIDs)
	m := g.CPU

	if B <= 1 || !gpu.BatchGEMMReady() {
		return nil // fall back to sequential
	}

	fmt.Printf("[prefill] batch=%d tokens, %d layers\n", B, len(g.Layers))

	// Allocate batch buffers: [B × dim]
	bHidden := gpu.NewDevBuf(B * h)
	bNormed := gpu.NewDevBuf(B * h)
	bQ := gpu.NewDevBuf(B * h)
	bK := gpu.NewDevBuf(B * kvDim)
	bV := gpu.NewDevBuf(B * kvDim)
	bAttnOut := gpu.NewDevBuf(B * h)
	bOOut := gpu.NewDevBuf(B * h)
	bGate := gpu.NewDevBuf(B * cfg.Intermediate)
	bUp := gpu.NewDevBuf(B * cfg.Intermediate)
	bDown := gpu.NewDevBuf(B * h)
	bResidual := gpu.NewDevBuf(B * h)

	// Embed all tokens into batch hidden: [B × h]
	embData := m.EmbedTokens.Data()
	hd := bHidden.Data()
	for i, tokID := range tokenIDs {
		copy(hd[i*h:(i+1)*h], embData[tokID*h:(tokID+1)*h])
	}
	bHidden.MarkDirty()
	bHidden.ToGPU()

	// Process each layer with batched ops
	for l := 0; l < len(g.Layers); l++ {
		layer := &g.Layers[l]

		// Sync every 20 layers to prevent command queue overflow
		if l > 0 && l%20 == 0 {
			gpu.Sync()
		}

		// Save residual: bResidual = bHidden
		gpu.DevCopy(bResidual, bHidden)

		// RMSNorm each row: bNormed[b] = rmsNorm(bHidden[b])
		// For now, do per-row RMSNorm on GPU (each row independently)
		for b := 0; b < B; b++ {
			hSlice := bHidden.Slice(b*h, h)
			nSlice := bNormed.Slice(b*h, h)
			gpu.DevRMSNorm(nSlice, hSlice, layer.InputNorm, float32(cfg.RMSNormEps))
		}

		// Batched Q/K/V projections: read weights once for all B tokens
		if layer.QWg != nil {
			gpu.PrefetchWeights(layer.OWg, layer.GateWg, layer.UpWg, layer.DownWg)
			gpu.GemmQ4(bQ, bNormed, layer.QWg, B)
			gpu.GemmQ4(bK, bNormed, layer.KWg, B)
			gpu.GemmQ4(bV, bNormed, layer.VWg, B)
			gpu.WaitPrefetch()
		} else {
			// CPU fallback for this layer — shouldn't happen for 7B
			return nil
		}

		// Bias (per-row broadcast)
		if layer.QB != nil {
			for b := 0; b < B; b++ {
				gpu.DevAdd(bQ.Slice(b*h, h), bQ.Slice(b*h, h), layer.QB)
				gpu.DevAdd(bK.Slice(b*kvDim, kvDim), bK.Slice(b*kvDim, kvDim), layer.KB)
				gpu.DevAdd(bV.Slice(b*kvDim, kvDim), bV.Slice(b*kvDim, kvDim), layer.VB)
			}
		}

		// RoPE + KV cache + Attention (per token — needs causal masking)
		for b := 0; b < B; b++ {
			pos := b
			seqLen := b + 1

			qSlice := bQ.Slice(b*h, h)
			kSlice := bK.Slice(b*kvDim, kvDim)
			vSlice := bV.Slice(b*kvDim, kvDim)

			// RoPE
			ropePtr := (*gpu.Buffer)(nil)
			if g.ropeCosSin != nil {
				ropePtr = g.ropeCosSin.GPUPtr()
			}
			if ropePtr != nil {
				gpu.DevRoPE(qSlice, g.ropeCosSin, pos, numHeads, headDim)
				gpu.DevRoPE(kSlice, g.ropeCosSin, pos, numKVHeads, headDim)
			} else {
				qd := qSlice.Data()
				kd := kSlice.Data()
				applyRoPE(qd, m.RopeFreqs, pos, numHeads, headDim)
				applyRoPE(kd, m.RopeFreqs, pos, numKVHeads, headDim)
				qSlice.MarkDirty()
				kSlice.MarkDirty()
			}

			// KV cache append
			var kvKPtr, kvVPtr, kPtr, vPtr *gpu.Buffer
			if g.kvGPU_K[l] != nil {
				kvKPtr = g.kvGPU_K[l].GPUPtr()
			}
			if g.kvGPU_V[l] != nil {
				kvVPtr = g.kvGPU_V[l].GPUPtr()
			}
			kPtr = kSlice.GPUPtr()
			vPtr = vSlice.GPUPtr()
			if kvKPtr != nil && kvVPtr != nil && kPtr != nil && vPtr != nil {
				kOff := gpu.CUdeviceptr(uint64(pos) * uint64(kvDim) * 4)
				_ = gpu.CopyDtoD(kvKPtr.Ptr+kOff, kPtr.Ptr, uint64(kvDim*4))
				_ = gpu.CopyDtoD(kvVPtr.Ptr+kOff, vPtr.Ptr, uint64(kvDim*4))
			}

			// Attention
			aSlice := bAttnOut.Slice(b*h, h)
			if g.kvGPU_K[l] != nil {
				gpu.DevAttention(aSlice, qSlice, g.kvGPU_K[l], g.kvGPU_V[l], seqLen, numHeads, numKVHeads, headDim, defaultScale)
			}
		}

		// Batched O projection
		if layer.OWg != nil {
			gpu.GemmQ4(bOOut, bAttnOut, layer.OWg, B)
		}

		// Residual add: bHidden = bResidual + bOOut
		gpu.DevAdd(bHidden, bResidual, bOOut)

		// Post-attention norm (per row)
		for b := 0; b < B; b++ {
			hSlice := bHidden.Slice(b*h, h)
			nSlice := bNormed.Slice(b*h, h)
			gpu.DevRMSNorm(nSlice, hSlice, layer.PostNorm, float32(cfg.RMSNormEps))
		}

		// Batched MLP
		if layer.GateWg != nil {
			gpu.PrefetchWeights(nil) // prefetch next layer if exists
			if l+1 < len(g.Layers) {
				next := &g.Layers[l+1]
				gpu.PrefetchWeights(next.QWg, next.KWg, next.VWg)
			}
			gpu.GemmQ4(bGate, bNormed, layer.GateWg, B)
			gpu.GemmQ4(bUp, bNormed, layer.UpWg, B)
		}

		// SiLU(gate) * up (per row)
		for b := 0; b < B; b++ {
			gSlice := bGate.Slice(b*cfg.Intermediate, cfg.Intermediate)
			uSlice := bUp.Slice(b*cfg.Intermediate, cfg.Intermediate)
			gpu.DevSiLUMul(gSlice, gSlice, uSlice)
		}

		// Batched down projection
		if layer.DownWg != nil {
			gpu.GemmQ4(bDown, bGate, layer.DownWg, B)
		}

		// Save residual for this stage
		gpu.DevCopy(bResidual, bHidden)

		// Residual add: bHidden = bResidual + bDown
		gpu.DevAdd(bHidden, bResidual, bDown)
	}

	// Extract last token's hidden state
	gpu.Sync()
	hd = bHidden.Data()
	lastHidden := make([]float32, h)
	copy(lastHidden, hd[(B-1)*h:B*h])

	fmt.Printf("[prefill] done, returning hidden[%d]\n", B-1)
	return lastHidden
}
