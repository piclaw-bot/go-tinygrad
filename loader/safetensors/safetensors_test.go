package safetensors

import (
	"math"
	"os"
	"sort"
	"testing"
)

func TestOpenGTESmall(t *testing.T) {
	path := gteSmallPath(t)

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	// Check we got reasonable number of tensors
	if len(f.Tensors) < 100 {
		t.Fatalf("expected >100 tensors, got %d", len(f.Tensors))
	}
	t.Logf("Loaded %d tensors", len(f.Tensors))

	// Check a known tensor
	info, ok := f.Tensors["embeddings.word_embeddings.weight"]
	if !ok {
		t.Fatal("missing embeddings.word_embeddings.weight")
	}
	if info.DType != "F16" {
		t.Fatalf("expected F16, got %s", info.DType)
	}
	if info.Shape[0] != 30522 || info.Shape[1] != 384 {
		t.Fatalf("unexpected shape: %v", info.Shape)
	}

	// Load and check values
	data, shape, err := f.GetFloat32("embeddings.word_embeddings.weight")
	if err != nil {
		t.Fatalf("GetFloat32: %v", err)
	}
	if len(data) != 30522*384 {
		t.Fatalf("expected %d values, got %d", 30522*384, len(data))
	}
	if shape[0] != 30522 || shape[1] != 384 {
		t.Fatalf("shape: %v", shape)
	}

	// Values should be reasonable floats (not NaN, not huge)
	hasNonZero := false
	for _, v := range data[:1000] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("bad value: %v", v)
		}
		if v != 0 {
			hasNonZero = true
		}
	}
	if !hasNonZero {
		t.Fatal("all zeros in first 1000 values")
	}
	t.Logf("First 5 values: %v", data[:5])
}

func gteSmallPath(t *testing.T) string {
	t.Helper()
	if path := os.Getenv("SAFETENSORS_PATH"); path != "" {
		if _, err := os.Stat(path); err == nil {
			return path
		}
		t.Skipf("model not found: %s", path)
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
	t.Skip("GTE-small safetensors fixture not found")
	return ""
}

func TestListTensors(t *testing.T) {
	path := gteSmallPath(t)

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	names := f.Names()
	sort.Strings(names)
	t.Logf("Tensor count: %d", len(names))
	for _, n := range names[:10] {
		info := f.Tensors[n]
		t.Logf("  %s: %s %v", n, info.DType, info.Shape)
	}
}

func TestFloat16Conversion(t *testing.T) {
	// Known F16 values
	tests := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},                     // +0
		{0x8000, -0},                    // -0 (bit pattern)
		{0x3C00, 1.0},                   // 1.0
		{0xC000, -2.0},                  // -2.0
		{0x3555, 0.333251953},           // ~1/3
		{0x7C00, float32(math.Inf(1))},  // +Inf
		{0xFC00, float32(math.Inf(-1))}, // -Inf
	}
	for _, tt := range tests {
		got := float16ToFloat32(tt.bits)
		if math.IsInf(float64(tt.want), 0) {
			if !math.IsInf(float64(got), 0) {
				t.Errorf("f16(0x%04x) = %v, want %v", tt.bits, got, tt.want)
			}
		} else if math.Abs(float64(got-tt.want)) > 0.001 {
			t.Errorf("f16(0x%04x) = %v, want %v", tt.bits, got, tt.want)
		}
	}
}
