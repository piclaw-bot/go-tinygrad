package model

import (
	"math"
	"os"
	"testing"
)

func TestLoadGTESmall(t *testing.T) {
	path := os.Getenv("SAFETENSORS_PATH")
	if path == "" {
		path = "../../gte-go/models/gte-small/model.safetensors"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not found: %s", path)
	}

	m, err := LoadGTESmall(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if m.Config.NumLayers != 12 {
		t.Fatalf("layers=%d want 12", m.Config.NumLayers)
	}
	if m.Config.HiddenSize != 384 {
		t.Fatalf("hidden=%d want 384", m.Config.HiddenSize)
	}
	t.Logf("Loaded GTE-small: %d layers, hidden=%d", m.Config.NumLayers, m.Config.HiddenSize)
}

func TestForwardGTESmall(t *testing.T) {
	path := os.Getenv("SAFETENSORS_PATH")
	if path == "" {
		path = "../../gte-go/models/gte-small/model.safetensors"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not found: %s", path)
	}

	m, err := LoadGTESmall(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Tokenize "I love cats" manually (CLS=101, I=1045, love=2293, cats=8870, SEP=102)
	tokenIDs := []int{101, 1045, 2293, 8870, 102}
	attnMask := []bool{true, true, true, true, true}

	emb := m.Embed(tokenIDs, attnMask)
	if len(emb) != 384 {
		t.Fatalf("embedding dim=%d want 384", len(emb))
	}

	// Check L2 normalized
	norm := float32(0)
	for _, v := range emb {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if math.Abs(float64(norm-1.0)) > 0.01 {
		t.Fatalf("norm=%v want ~1.0", norm)
	}

	// Check not all zeros
	nonZero := 0
	for _, v := range emb {
		if v != 0 {
			nonZero++
		}
	}
	if nonZero < 100 {
		t.Fatalf("too many zeros: %d non-zero out of 384", nonZero)
	}

	t.Logf("Embedding[0:5]: %v", emb[:5])
	t.Logf("Norm: %v, non-zero: %d/384", norm, nonZero)
}
