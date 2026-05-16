package gpu

import (
	"sync"
	"unsafe"
)

// --- GPU RoPE + Attention ---

var (
	ropeOnce         sync.Once
	ropeFn           CUfunction
	ropePartialFn    CUfunction
	attnScoreFn      CUfunction
	softmaxRowsFn    CUfunction
	attnFn           CUfunction
	ropeReady        bool
	ropePartialReady bool
	attnScoreReady   bool
	softmaxRowsReady bool
	attnReady        bool
)

func initRoPEAttn() { loadMegaModule() }

// DevRoPE applies rotary position embedding on GPU (in-place).
// cosSin is a precomputed [maxSeq * headDim] buffer with interleaved cos,sin pairs.
func DevRoPE(x *DevBuf, cosSin *DevBuf, pos, nHeads, headDim int) bool {
	initRoPEAttn()
	total, okTotal := checkedMulInt(nHeads, headDim)
	halfDim, okHalf := checkedMulInt(nHeads, headDim/2)
	cosNeed, okCosNeed := checkedMulInt(pos+1, headDim)
	if ropeReady && fitsUint32(pos) && nHeads > 0 && headDim > 0 && headDim%2 == 0 && okTotal && okHalf && okCosNeed && x != nil && cosSin != nil && x.n >= total && cosSin.n >= cosNeed && tryGPU(x, cosSin) {
		grid, okGrid := grid1DFor(halfDim, 256)
		if !okGrid {
			return false
		}
		p := uint32(pos)
		nh := uint32(nHeads)
		hd := uint32(headDim)
		if err := LaunchKernel(ropeFn, grid, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&x.gpu.Ptr),
			unsafe.Pointer(&cosSin.gpu.Ptr),
			unsafe.Pointer(&p),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&hd)); err == nil {
			x.dev = GPU_DEVICE
			return true
		}
	}
	return false
}

// DevRoPEPartial applies partial rotary position embedding on GPU (in-place).
// cosSin is [maxSeq * rotHalf * 2] with interleaved cos,sin pairs.
func DevRoPEPartial(x *DevBuf, cosSin *DevBuf, pos, nHeads, headDim, rotHalf int) bool {
	initRoPEAttn()
	total, okTotal := checkedMulInt(nHeads, headDim)
	totalPairs, okPairs := checkedMulInt(nHeads, rotHalf)
	posPairs, okPosPairs := checkedMulInt(pos+1, rotHalf)
	cosNeed, okCosNeed := checkedMulInt(posPairs, 2)
	if ropePartialReady && fitsUint32(pos) && nHeads > 0 && headDim > 0 && rotHalf > 0 && rotHalf <= headDim/2 && okTotal && okPairs && okPosPairs && okCosNeed && x != nil && cosSin != nil && x.n >= total && cosSin.n >= cosNeed && tryGPU(x, cosSin) {
		grid, okGrid := grid1DFor(totalPairs, 256)
		if !okGrid {
			return false
		}
		p := uint32(pos)
		nh := uint32(nHeads)
		hd := uint32(headDim)
		rh := uint32(rotHalf)
		if err := LaunchKernel(ropePartialFn, grid, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&x.gpu.Ptr),
			unsafe.Pointer(&cosSin.gpu.Ptr),
			unsafe.Pointer(&p),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&rh)); err == nil {
			x.dev = GPU_DEVICE
			return true
		}
	}
	return false
}

// DevAttentionScores runs the score phase of GQA attention on GPU.
// out[nHeads*seqLen], q[nHeads*headDim], kCache[seqLen*kvDim]
func DevAttentionScores(out, q, kCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) bool {
	initRoPEAttn()
	qLen, okQ := checkedMulInt(nHeads, headDim)
	kvDim, okKVDim := checkedMulInt(nKVHeads, headDim)
	cacheLen, okCache := checkedMulInt(seqLen, kvDim)
	scoreLen, okScore := checkedMulInt(nHeads, seqLen)
	if attnScoreReady && fitsUint32(seqLen) && seqLen > 0 && seqLen <= 2048 && fitsUint32(nHeads) && fitsUint32(nKVHeads) && fitsUint32(headDim) && nHeads > 0 && nKVHeads > 0 && headDim > 0 && okQ && okKVDim && okCache && okScore && out != nil && q != nil && kCache != nil && out.n >= scoreLen && q.n >= qLen && kCache.n >= cacheLen && tryGPU(out, q, kCache) {
		sl := uint32(seqLen)
		nh := uint32(nHeads)
		nkv := uint32(nKVHeads)
		hd := uint32(headDim)
		if err := LaunchKernel(attnScoreFn, uint32(nHeads), 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&q.gpu.Ptr),
			unsafe.Pointer(&kCache.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&nkv),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&scale)); err == nil {
			out.dev = GPU_DEVICE
			return true
		}
	}
	return false
}

// DevSoftmaxRows runs the softmax phase over contiguous score rows.
// in/out[nRows*seqLen], one block per row.
func DevSoftmaxRows(out, in *DevBuf, nRows, seqLen int) bool {
	initRoPEAttn()
	total, okTotal := checkedMulInt(nRows, seqLen)
	if softmaxRowsReady && fitsUint32(nRows) && fitsUint32(seqLen) && nRows > 0 && seqLen > 0 && seqLen <= 2048 && okTotal && out != nil && in != nil && out.n >= total && in.n >= total && tryGPU(out, in) {
		sl := uint32(seqLen)
		if err := LaunchKernel(softmaxRowsFn, uint32(nRows), 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&in.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl)); err == nil {
			out.dev = GPU_DEVICE
			return true
		}
	}
	return false
}

// DevAttention runs GQA attention on GPU.
// q[nHeads*headDim], kCache/vCache[seqLen*kvDim], out[nHeads*headDim]
func DevAttention(out, q, kCache, vCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) {
	initRoPEAttn()
	qLen, okQ := checkedMulInt(nHeads, headDim)
	kvDim, okKVDim := checkedMulInt(nKVHeads, headDim)
	cacheLen, okCache := checkedMulInt(seqLen, kvDim)
	if attnReady && fitsUint32(seqLen) && seqLen > 0 && seqLen <= 2048 && fitsUint32(nHeads) && fitsUint32(nKVHeads) && fitsUint32(headDim) && nHeads > 0 && nKVHeads > 0 && headDim > 0 && okQ && okKVDim && okCache && out != nil && q != nil && kCache != nil && vCache != nil && out.n >= qLen && q.n >= qLen && kCache.n >= cacheLen && vCache.n >= cacheLen && tryGPU(out, q, kCache, vCache) {
		sl := uint32(seqLen)
		nh := uint32(nHeads)
		nkv := uint32(nKVHeads)
		hd := uint32(headDim)
		// One block per query head, 256 threads per block
		if err := LaunchKernel(attnFn, uint32(nHeads), 1, 1, 256, 1, 1, 2048*4,
			unsafe.Pointer(&q.gpu.Ptr),
			unsafe.Pointer(&kCache.gpu.Ptr),
			unsafe.Pointer(&vCache.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&nkv),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&scale)); err == nil {
			out.dev = GPU_DEVICE
		}
		return
	}
	// CPU fallback in model code
}
