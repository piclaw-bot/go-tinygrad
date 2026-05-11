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
	assertCloseFloat32Slice(t, "allocated", got, want, 2e-5)

	gotScratch := make([]float32, numHeads*headDim)
	scores := make([]float32, seqLen)
	for i := range gotScratch {
		gotScratch[i] = 123 // ensure gqaAttentionScaleInto clears reusable output
	}
	gqaAttentionScaleInto(gotScratch, scores, q, k, v, seqLen, numHeads, numKVHeads, headDim, scale)
	assertCloseFloat32Slice(t, "scratch", gotScratch, want, 2e-5)
}

func assertCloseFloat32Slice(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("%s[%d]=%.8f want %.8f diff %.8g", name, i, got[i], want[i], diff)
		}
	}
}

func TestAttentionMalformedInputsDoNotPanic(t *testing.T) {
	if got := gqaAttention(nil, nil, nil, 1, 1, 0, 0); got != nil {
		t.Fatalf("gqaAttention malformed=%v, want nil", got)
	}
	out := []float32{99, 100}
	scores := []float32{1}
	gqaAttentionScaleInto(out, scores, nil, nil, nil, 1, 2, 0, 1, 1)
	if out[0] != 99 || out[1] != 100 {
		t.Fatalf("malformed attention modified out: %v", out)
	}
	got := gqaAttentionScale(nil, nil, nil, 0, 2, 1, 2, 1)
	if len(got) != 4 {
		t.Fatalf("zero-seq attention len=%d want 4", len(got))
	}
}

func TestRoPEPartialMalformedInputsDoNotPanic(t *testing.T) {
	x := []float32{1, 2}
	applyRoPEPartial(x, nil, 0, 1, 2, 1)
	applyRoPEPartial(x, []float32{1, 0}, -1, 1, 2, 1)
	applyRoPEPartial(x, []float32{1, 0}, 0, 4, 2, 99)
}
