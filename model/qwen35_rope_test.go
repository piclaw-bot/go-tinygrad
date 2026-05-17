package model

import "testing"

func TestQwen35RotaryHalf(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.HeadDim = 256
	meta.PartialRotaryFactor = 0.25
	if got := Qwen35RotaryHalf(meta); got != 32 {
		t.Fatalf("rotHalf=%d want 32", got)
	}
	freqs := NewQwen35RoPEFreqs(meta, 4)
	if len(freqs) != 4*32*2 {
		t.Fatalf("freq len=%d", len(freqs))
	}
}
