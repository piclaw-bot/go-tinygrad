package model

import (
	"math"
	"sync"

	"github.com/rcarmo/go-pherence/runtime/quant"

	"github.com/rcarmo/go-pherence/backends/simd"
	"github.com/rcarmo/go-pherence/gpu"
)

func uploadExpertNativeToPool(pool *gpu.ExpertPool, layer *LlamaLayer, expertID, poolKey, moeInter, hidden int) *gpu.ExpertEntry {
	if pool == nil || layer == nil || expertID < 0 || expertID >= len(layer.ExpertGateW) || expertID >= len(layer.ExpertUpW) || expertID >= len(layer.ExpertDownW) {
		return nil
	}
	if layer.ExpertGateW[expertID] == nil || layer.ExpertUpW[expertID] == nil || layer.ExpertDownW[expertID] == nil {
		return nil
	}
	entry := &gpu.ExpertEntry{ExpertID: poolKey}
	ew := layer.ExpertGateW[expertID]
	gw, err1 := gpu.UploadMLXWeightNative(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize)
	ew = layer.ExpertUpW[expertID]
	uw, err2 := gpu.UploadMLXWeightNative(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize)
	ew = layer.ExpertDownW[expertID]
	dw, err3 := gpu.UploadMLXWeightNative(ew.Weight, ew.Scales, ew.Biases, ew.InDim, ew.OutDim, ew.GroupSize)
	if err1 != nil || err2 != nil || err3 != nil {
		gpu.FreeExpertEntry(&gpu.ExpertEntry{GateW: gw, UpW: uw, DownW: dw})
		return nil
	}
	entry.GateW = gw
	entry.UpW = uw
	entry.DownW = dw
	entry.SizeBytes = int64(3 * moeInter * hidden / 2)
	if evicted := pool.Put(entry); evicted != nil {
		gpu.FreeExpertEntry(evicted)
	}
	return entry
}

// moeForwardGPU runs the MoE forward pass using GPU for hot experts.
// Falls back to CPU quant.GemvMLQ for cold experts not in the pool.
func moeForwardGPU(outDev, xDev *gpu.DevBuf, layer *LlamaLayer, cfg LlamaConfig, pool *gpu.ExpertPool, layerIdx int, routerGPU *gpu.GPUMLXWeight) []float32 {
	if layer == nil || xDev == nil || cfg.HiddenSize <= 0 || cfg.NumExperts <= 0 || cfg.MoEIntermediate <= 0 || xDev.Len() < cfg.HiddenSize {
		return nil
	}
	h := cfg.HiddenSize
	if outDev != nil && outDev.Len() < h {
		outDev = nil
	}
	var xCPU []float32
	getXCPU := func() []float32 {
		if xCPU == nil && xDev != nil {
			xCPU = append([]float32(nil), xDev.Data()[:h]...)
		}
		return xCPU
	}
	numExperts := cfg.NumExperts
	numActive := cfg.NumExpertsPerTok
	if numActive <= 0 {
		numActive = 8
	}
	if numActive > numExperts {
		numActive = numExperts
	}

	// Router: compute logits for each expert.
	routerLogits := make([]float32, numExperts)
	if routerGPU != nil {
		routerOut, err := gpu.NewDevBufGPU(numExperts)
		if err == nil && xDev != nil {
			gpu.GemvMLXDirect(routerOut, xDev, routerGPU)
			copy(routerLogits, routerOut.Data()[:numExperts])
		} else if layer.RouterW != nil {
			quant.GemvMLQ(routerLogits, getXCPU(), layer.RouterW)
		}
		if routerOut != nil {
			routerOut.Free()
		}
	} else if layer.RouterW != nil {
		quant.GemvMLQ(routerLogits, getXCPU(), layer.RouterW)
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
	var xBuf, gateBuf, upBuf, downBuf, gpuOutBuf *gpu.DevBuf
	hasGPUExperts := pool != nil && pool.Slots() > 0 && len(selected) > 0
	if hasGPUExperts {
		xBuf = xDev
		var err error
		gateBuf, err = gpu.NewDevBufGPU(moeInter)
		if err != nil {
			hasGPUExperts = false
		}
		if hasGPUExperts {
			upBuf, err = gpu.NewDevBufGPU(moeInter)
			hasGPUExperts = err == nil
		}
		if hasGPUExperts {
			downBuf, err = gpu.NewDevBufGPU(h)
			hasGPUExperts = err == nil
		}
		if hasGPUExperts {
			gpuOutBuf, err = gpu.NewDevBufGPU(h)
			hasGPUExperts = err == nil
		}
		if !hasGPUExperts {
			gateBuf.Free()
			upBuf.Free()
			downBuf.Free()
			gpuOutBuf.Free()
			xBuf, gateBuf, upBuf, downBuf, gpuOutBuf = nil, nil, nil, nil, nil
		} else {
			defer gateBuf.Free()
			defer upBuf.Free()
			defer downBuf.Free()
			defer gpuOutBuf.Free()
		}
	}

	// Run CPU experts in parallel, GPU experts sequentially (shared buffers)
	type expertResult struct {
		down   []float32
		weight float32
	}
	results := make([]expertResult, len(selected))
	cpuFallbackUsed := false
	var wg sync.WaitGroup
	type uploadCandidate struct {
		expertID int
		poolKey  int
	}
	uploadAfterCPU := make([]uploadCandidate, 0, len(selected))
	gpuOutInitialized := false

	for si, exp := range selected {
		eid := exp.id
		if eid < 0 || eid >= len(layer.ExpertGateW) || eid >= len(layer.ExpertUpW) || eid >= len(layer.ExpertDownW) || layer.ExpertGateW[eid] == nil || layer.ExpertUpW[eid] == nil || layer.ExpertDownW[eid] == nil {
			continue
		}

		poolKey := layerIdx*cfg.NumExperts + eid
		var gpuEntry *gpu.ExpertEntry
		if pool != nil {
			gpuEntry = pool.Get(poolKey)
			if gpuEntry == nil && xBuf != nil {
				gpuEntry = uploadExpertNativeToPool(pool, layer, eid, poolKey, moeInter, h)
			}
		}

		if gpuEntry != nil && gpuEntry.GateW != nil && xBuf != nil && gpuOutBuf != nil {
			// GPU path: reuse pre-allocated buffers
			gpu.GemvMLXDirect(gateBuf, xBuf, gpuEntry.GateW)
			gpu.GemvMLXDirect(upBuf, xBuf, gpuEntry.UpW)
			gpu.DevSiLUMul(gateBuf, gateBuf, upBuf)
			gpu.GemvMLXDirect(downBuf, gateBuf, gpuEntry.DownW)
			if gpuOutInitialized {
				gpu.DevAddScaled(gpuOutBuf, gpuOutBuf, downBuf, exp.score)
			} else {
				gpu.DevScale(gpuOutBuf, downBuf, exp.score)
				gpuOutInitialized = true
			}
		} else {
			cpuFallbackUsed = true
			// CPU fallback (parallel). CUDA uploads are deliberately deferred until
			// after wg.Wait(); the CUDA driver context is thread-local and the expert
			// pool may evict GPU buffers, so doing this inside worker goroutines can
			// race with in-flight GPU work.
			if pool != nil && gpuEntry == nil {
				uploadAfterCPU = append(uploadAfterCPU, uploadCandidate{expertID: eid, poolKey: poolKey})
			}
			xForCPU := getXCPU()
			wg.Add(1)
			go func(idx int, expertID int, w float32, xIn []float32) {
				defer wg.Done()
				gate := make([]float32, moeInter)
				up := make([]float32, moeInter)
				quant.GemvMLQ(gate, xIn, layer.ExpertGateW[expertID])
				quant.GemvMLQ(up, xIn, layer.ExpertUpW[expertID])
				simd.VecSiLUMul(gate, gate, up)
				down := make([]float32, h)
				quant.GemvMLQ(down, gate, layer.ExpertDownW[expertID])
				results[idx] = expertResult{down: down, weight: w}
			}(si, eid, exp.score, xForCPU)
		}
	}
	wg.Wait()

	if gpuOutBuf != nil && gpuOutInitialized {
		if outDev != nil {
			gpu.DevCopy(outDev, gpuOutBuf)
		} else {
			gpuOut := gpuOutBuf.Data()
			for i := range out {
				out[i] += gpuOut[i]
			}
		}
	}

	// Warm the GPU expert pool sequentially after CPU fallback work completes.
	for _, cand := range uploadAfterCPU {
		if pool.Peek(cand.poolKey) != nil {
			continue
		}
		uploadExpertNativeToPool(pool, layer, cand.expertID, cand.poolKey, moeInter, h)
	}

	for _, r := range results {
		if r.down == nil {
			continue
		}
		for i := range out {
			out[i] += r.weight * r.down[i]
		}
	}
	if outDev != nil && gpuOutInitialized {
		if cpuFallbackUsed {
			cpuOut := gpu.NewDevBufFrom(out)
			if cpuOut.ToGPU() == nil {
				gpu.DevAdd(outDev, outDev, cpuOut)
				cpuOut.Free()
				return nil
			}
			cpuOut.Free()
			// If re-uploading the CPU fallback contribution fails, return a complete
			// CPU result rather than dropping the GPU experts already accumulated in
			// gpuOutBuf. The caller will copy this back into the model buffer.
			gpuOut := gpuOutBuf.Data()
			for i := range out {
				out[i] += gpuOut[i]
			}
			return out
		}
		return nil
	}
	return out
}
