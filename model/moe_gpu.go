package model

import (
	"math"
	"sync"

	"github.com/rcarmo/go-pherence/backends/simd"
	"github.com/rcarmo/go-pherence/gpu"
)

// moeForwardGPU runs the MoE forward pass using GPU for hot experts.
// Falls back to CPU GemvMLQ for cold experts not in the pool.
func moeForwardGPU(x []float32, layer *LlamaLayer, cfg LlamaConfig, pool *gpu.ExpertPool, layerIdx int) []float32 {
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

	// Separate GPU-cached and CPU-fallback experts
	moeInter := cfg.MoEIntermediate
	out := make([]float32, h)

	// Pre-allocate GPU work buffers once (reused across experts)
	var xBuf, gateBuf, upBuf, downBuf *gpu.DevBuf
	hasGPUExperts := false
	for _, exp := range selected {
		poolKey := layerIdx*cfg.NumExperts + exp.id
		if pool != nil && pool.Get(poolKey) != nil {
			hasGPUExperts = true
			break
		}
	}
	if hasGPUExperts {
		xBuf = gpu.NewDevBufFrom(append([]float32(nil), x...))
		gateBuf = gpu.NewDevBuf(moeInter)
		upBuf = gpu.NewDevBuf(moeInter)
		downBuf = gpu.NewDevBuf(h)
		xBuf.ToGPU()
		gateBuf.ToGPU()
		upBuf.ToGPU()
		downBuf.ToGPU()
		defer xBuf.Free()
		defer gateBuf.Free()
		defer upBuf.Free()
		defer downBuf.Free()
	}

	// Run CPU experts in parallel, GPU experts sequentially (shared buffers)
	type expertResult struct {
		down   []float32
		weight float32
	}
	results := make([]expertResult, len(selected))
	var wg sync.WaitGroup
	type uploadCandidate struct {
		expertID int
		poolKey  int
	}
	uploadAfterCPU := make([]uploadCandidate, 0, len(selected))

	for si, exp := range selected {
		eid := exp.id
		if eid >= len(layer.ExpertGateW) || layer.ExpertGateW[eid] == nil {
			continue
		}

		poolKey := layerIdx*cfg.NumExperts + eid
		var gpuEntry *gpu.ExpertEntry
		if pool != nil {
			gpuEntry = pool.Get(poolKey)
		}

		if gpuEntry != nil && gpuEntry.GateW != nil && xBuf != nil {
			// GPU path: reuse pre-allocated buffers
			gpu.GemvMLXDirect(gateBuf, xBuf, gpuEntry.GateW)
			gpu.GemvMLXDirect(upBuf, xBuf, gpuEntry.UpW)
			gpu.DevSiLUMul(gateBuf, gateBuf, upBuf)
			gpu.GemvMLXDirect(downBuf, gateBuf, gpuEntry.DownW)
			gpu.Sync()
			results[si] = expertResult{
				down:   append([]float32(nil), downBuf.Data()[:h]...),
				weight: exp.score,
			}
		} else {
			// CPU fallback (parallel). CUDA uploads are deliberately deferred until
			// after wg.Wait(); the CUDA driver context is thread-local and the expert
			// pool may evict GPU buffers, so doing this inside worker goroutines can
			// race with in-flight GPU work.
			if pool != nil {
				uploadAfterCPU = append(uploadAfterCPU, uploadCandidate{expertID: eid, poolKey: poolKey})
			}
			wg.Add(1)
			go func(idx int, expertID int, w float32) {
				defer wg.Done()
				gate := make([]float32, moeInter)
				up := make([]float32, moeInter)
				GemvMLQ(gate, x, layer.ExpertGateW[expertID])
				GemvMLQ(up, x, layer.ExpertUpW[expertID])
				simd.VecSiLUMul(gate, gate, up)
				down := make([]float32, h)
				GemvMLQ(down, gate, layer.ExpertDownW[expertID])
				results[idx] = expertResult{down: down, weight: w}
			}(si, eid, exp.score)
		}
	}
	wg.Wait()

	// Warm the GPU expert pool sequentially after CPU fallback work completes.
	for _, cand := range uploadAfterCPU {
		if pool.Get(cand.poolKey) != nil {
			continue
		}
		entry := &gpu.ExpertEntry{ExpertID: cand.poolKey}
		ew := layer.ExpertGateW[cand.expertID]
		gw, err1 := gpu.UploadMLXWeight(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize, true)
		ew = layer.ExpertUpW[cand.expertID]
		uw, err2 := gpu.UploadMLXWeight(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize, true)
		ew = layer.ExpertDownW[cand.expertID]
		dw, err3 := gpu.UploadMLXWeight(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize, true)
		if err1 == nil && err2 == nil && err3 == nil {
			entry.GateW = gw
			entry.UpW = uw
			entry.DownW = dw
			entry.SizeBytes = int64(3 * moeInter * h / 2)
			evicted := pool.Put(entry)
			if evicted != nil {
				gpu.FreeExpertEntry(evicted)
			}
		}
	}

	for _, r := range results {
		if r.down == nil {
			continue
		}
		for i := range out {
			out[i] += r.weight * r.down[i]
		}
	}

	return out
}
