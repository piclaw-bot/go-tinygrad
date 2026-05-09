package model

import (
	"math"

	"github.com/rcarmo/go-pherence/gpu"
)

// moeForwardGPU runs the MoE forward pass using GPU for hot experts.
// Falls back to CPU GemvMLQ for cold experts not in the pool.
func moeForwardGPU(x []float32, layer *LlamaLayer, cfg LlamaConfig, pool *gpu.ExpertPool) []float32 {
	h := len(x)
	numExperts := cfg.NumExperts
	numActive := cfg.NumExpertsPerTok
	if numActive <= 0 {
		numActive = 8
	}

	// Router: compute logits for each expert
	routerLogits := make([]float32, numExperts)
	if layer.RouterW != nil {
		GemvMLQ(routerLogits, x, layer.RouterW)
	}

	// Softmax over router logits
	maxLogit := routerLogits[0]
	for _, v := range routerLogits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var expSum float32
	for i := range routerLogits {
		routerLogits[i] = float32(math.Exp(float64(routerLogits[i] - maxLogit)))
		expSum += routerLogits[i]
	}
	for i := range routerLogits {
		routerLogits[i] /= expSum
	}

	// Top-k selection
	type expertScore struct {
		id    int
		score float32
	}
	selected := make([]expertScore, 0, numActive)
	for i := 0; i < numActive; i++ {
		bestID := -1
		bestScore := float32(-1)
		for j, s := range routerLogits {
			if s > bestScore {
				alreadyPicked := false
				for _, sel := range selected {
					if sel.id == j {
						alreadyPicked = true
						break
					}
				}
				if !alreadyPicked {
					bestID = j
					bestScore = s
				}
			}
		}
		if bestID >= 0 {
			selected = append(selected, expertScore{id: bestID, score: bestScore})
		}
	}

	// Normalize selected weights
	if cfg.NormTopKProb {
		var sum float32
		for _, s := range selected {
			sum += s.score
		}
		if sum > 0 {
			for i := range selected {
				selected[i].score /= sum
			}
		}
	}

	// Run selected experts — GPU for cached, CPU for cold
	moeInter := cfg.MoEIntermediate
	out := make([]float32, h)

	for _, exp := range selected {
		eid := exp.id
		if eid >= len(layer.ExpertGateW) || layer.ExpertGateW[eid] == nil {
			continue
		}

		// Check GPU expert pool
		var gpuEntry *gpu.ExpertEntry
		if pool != nil {
			gpuEntry = pool.Get(eid)
		}

		var down []float32
		if gpuEntry != nil && gpuEntry.GateW != nil {
			// GPU path: run expert MLP on GPU
			xBuf := gpu.NewDevBufFrom(append([]float32(nil), x...))
			gateBuf := gpu.NewDevBuf(moeInter)
			upBuf := gpu.NewDevBuf(moeInter)
			downBuf := gpu.NewDevBuf(h)
			xBuf.ToGPU()
			gateBuf.ToGPU()
			upBuf.ToGPU()
			downBuf.ToGPU()

			gpu.GemvMLXDirect(gateBuf, xBuf, gpuEntry.GateW)
			gpu.GemvMLXDirect(upBuf, xBuf, gpuEntry.UpW)
			gpu.DevSiLUMul(gateBuf, gateBuf, upBuf)
			gpu.GemvMLXDirect(downBuf, gateBuf, gpuEntry.DownW)
			gpu.Sync()

			down = append([]float32(nil), downBuf.Data()[:h]...)
			xBuf.Free()
			gateBuf.Free()
			upBuf.Free()
			downBuf.Free()
		} else {
			// CPU fallback
			gate := make([]float32, moeInter)
			up := make([]float32, moeInter)
			GemvMLQ(gate, x, layer.ExpertGateW[eid])
			GemvMLQ(up, x, layer.ExpertUpW[eid])
			for i := range gate {
				sig := float32(1.0 / (1.0 + math.Exp(float64(-gate[i]))))
				gate[i] = gate[i] * sig * up[i]
			}
			down = make([]float32, h)
			GemvMLQ(down, gate, layer.ExpertDownW[eid])

			// Upload to expert pool for next time
			if pool != nil {
				entry := &gpu.ExpertEntry{ExpertID: eid}
				ew := layer.ExpertGateW[eid]
				gw, err1 := gpu.UploadMLXWeight(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize, true)
				ew = layer.ExpertUpW[eid]
				uw, err2 := gpu.UploadMLXWeight(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize, true)
				ew = layer.ExpertDownW[eid]
				dw, err3 := gpu.UploadMLXWeight(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize, true)
				if err1 == nil && err2 == nil && err3 == nil {
					entry.GateW = gw
					entry.UpW = uw
					entry.DownW = dw
					entry.SizeBytes = int64(3 * moeInter * h / 2) // approximate
					evicted := pool.Put(entry)
					if evicted != nil {
						gpu.FreeExpertEntry(evicted)
					}
				}
			}
		}

		// Weighted accumulation
		w := exp.score
		for i := range out {
			out[i] += w * down[i]
		}
	}

	return out
}
