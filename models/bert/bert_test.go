package bert

import (
	"math"
	"os"
	"testing"
)

func TestLoadGTESmall(t *testing.T) {
	path := gteSmallPath(t)

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
	path := gteSmallPath(t)

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

func gteSmallPath(tb testing.TB) string {
	tb.Helper()
	if path := os.Getenv("SAFETENSORS_PATH"); path != "" {
		if _, err := os.Stat(path); err == nil {
			return path
		}
		tb.Skipf("model not found: %s", path)
	}
	for _, path := range []string{
		"../../../gte-go/models/gte-small/model.safetensors",
		"../../gte-go/models/gte-small/model.safetensors",
		"../gte-go/models/gte-small/model.safetensors",
	} {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	tb.Skip("GTE-small safetensors fixture not found")
	return ""
}

func BenchmarkGTESmallEmbed(b *testing.B) {
	path := gteSmallPath(b)
	m, err := LoadGTESmall(path)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	tokenIDs := []int{101, 1045, 2293, 8870, 102} // "I love cats"
	attnMask := []bool{true, true, true, true, true}

	// Warmup
	for i := 0; i < 3; i++ {
		_ = m.Embed(tokenIDs, attnMask)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.Embed(tokenIDs, attnMask)
	}
}

func TestForwardFastCorrectness(t *testing.T) {
	path := gteSmallPath(t)
	m, err := LoadGTESmall(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tokenIDs := []int{101, 1045, 2293, 8870, 102}
	attnMask := []bool{true, true, true, true, true}

	slow := m.Embed(tokenIDs, attnMask)
	fast := m.EmbedFast(tokenIDs, attnMask)

	maxDiff := float32(0)
	for i := range slow {
		d := slow[i] - fast[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 0.001 {
		t.Fatalf("fast vs slow maxDiff=%v (too large)", maxDiff)
	}
	t.Logf("fast vs slow maxDiff=%v", maxDiff)
}

func BenchmarkGTESmallEmbedFast(b *testing.B) {
	path := gteSmallPath(b)
	m, err := LoadGTESmall(path)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	tokenIDs := []int{101, 1045, 2293, 8870, 102}
	attnMask := []bool{true, true, true, true, true}
	for i := 0; i < 3; i++ {
		_ = m.EmbedFast(tokenIDs, attnMask)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.EmbedFast(tokenIDs, attnMask)
	}
}
