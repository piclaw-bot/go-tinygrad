package model

import (
	"math"
	"testing"
)

func gqaAttentionScaleReference(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	h := numHeads * headDim
	kvDim := numKVHeads * headDim
	headsPerKV := numHeads / numKVHeads
	out := make([]float32, h)
	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV
		scores := make([]float32, seqLen)
		for t := 0; t < seqLen; t++ {
			sum := float32(0)
			for d := 0; d < headDim; d++ {
				sum += q[head*headDim+d] * kCache[t*kvDim+kvHead*headDim+d]
			}
			scores[t] = sum * scale
		}
		mx := scores[0]
		for _, v := range scores[1:] {
			if v > mx {
				mx = v
			}
		}
		expSum := float32(0)
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - mx)))
			expSum += scores[i]
		}
		for i := range scores {
			scores[i] /= expSum
		}
		for d := 0; d < headDim; d++ {
			sum := float32(0)
			for t := 0; t < seqLen; t++ {
				sum += scores[t] * vCache[t*kvDim+kvHead*headDim+d]
			}
			out[head*headDim+d] = sum
		}
	}
	return out
}

func TestGQAAttentionScaleMatchesReference(t *testing.T) {
	seqLen := 17
	numHeads := 6
	numKVHeads := 2
	headDim := 32
	q := benchSeq(numHeads * headDim)
	k := benchSeq(seqLen * numKVHeads * headDim)
	v := benchSeq(seqLen * numKVHeads * headDim)
	scale := float32(0.37)
	got := gqaAttentionScale(q, k, v, seqLen, numHeads, numKVHeads, headDim, scale)
	want := gqaAttentionScaleReference(q, k, v, seqLen, numHeads, numKVHeads, headDim, scale)
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 2e-5 {
			t.Fatalf("out[%d]=%.8f want %.8f diff %.8g", i, got[i], want[i], diff)
		}
	}
}
