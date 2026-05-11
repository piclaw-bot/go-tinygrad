package safetensors

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/memory"
)

func TestMmapAdvisorWithFile(t *testing.T) {
	// Use an actual safetensors file if available.
	f := openOptionalSafetensors(t, smolLM2Candidates())
	defer f.Close()

	if f.mmapData == nil {
		t.Skip("file not mmap'd")
	}

	a := memory.NewMmapAdvisor(f.mmapData)

	// Simulate layer-by-layer access pattern.
	for name, info := range f.Tensors {
		off := int64(info.DataOffsets[0]) + int64(8+f.headerSize) // absolute offset in mmap
		sz := int64(info.DataOffsets[1] - info.DataOffsets[0])
		a.Prefetch(off, sz)
		_ = name
	}

	nRanges, hotBytes, peakBytes := a.Stats()
	t.Logf("after prefetching all tensors: %d ranges, %.2f MB hot, %.2f MB peak",
		nRanges, float64(hotBytes)/(1024*1024), float64(peakBytes)/(1024*1024))

	a.MergeRanges()
	nMerged, _, _ := a.Stats()
	t.Logf("after merge: %d ranges (was %d)", nMerged, nRanges)
}

func openOptionalSafetensors(t *testing.T, candidates []string) *File {
	t.Helper()
	var lastErr error
	for _, path := range candidates {
		f, err := Open(path)
		if err == nil {
			return f
		}
		lastErr = err
	}
	t.Skipf("model not found: %v", lastErr)
	return nil
}

func smolLM2Candidates() []string {
	var out []string
	if dir := os.Getenv("SMOLLM_PATH"); dir != "" {
		out = append(out, filepath.Join(dir, "model.safetensors"))
	}
	out = append(out,
		"../../models/smollm2-135m/model.safetensors",
		"../../../models/smollm2-135m/model.safetensors",
		"../models/smollm2-135m/model.safetensors",
	)
	return out
}

func TestEagerLoadWithFile(t *testing.T) {
	f := openOptionalSafetensors(t, smolLM2Candidates())
	defer f.Close()

	bytes, err := f.EagerLoad()
	if err != nil {
		t.Fatalf("EagerLoad: %v", err)
	}
	if bytes != int64(len(f.mmapData)) {
		t.Fatalf("EagerLoad bytes=%d want %d", bytes, len(f.mmapData))
	}
	nRanges, hotBytes, peakBytes := f.Advisor.Stats()
	t.Logf("eager loaded %.2f MB: ranges=%d hot=%.2f MB peak=%.2f MB",
		float64(bytes)/(1024*1024), nRanges, float64(hotBytes)/(1024*1024), float64(peakBytes)/(1024*1024))
	if nRanges == 0 || hotBytes == 0 || peakBytes == 0 {
		t.Fatal("expected eager load to update advisor stats")
	}
}
