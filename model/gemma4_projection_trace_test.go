package model

import (
	"math"
	"os"
	"sort"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

type opTraceKey struct {
	layer int
	op    string
}

func topOps(m map[opTraceKey][]float32) []opTraceKey {
	keys := make([]opTraceKey, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		return keys[i].op < keys[j].op
	})
	return keys
}

func TestGemma4CPUvsGPUProjectionTrace(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 CPU/GPU projection trace")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}
	t.Cleanup(gpu.Shutdown)

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1
	targetLayers := map[int]bool{0: true, 14: true, 15: true, 34: true}
	cpuOps := map[opTraceKey][]float32{}
	gpuOps := map[opTraceKey][]float32{}

	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if step != traceStep || !targetLayers[layer] {
			return
		}
		cp := append([]float32(nil), vec...)
		k := opTraceKey{layer: layer, op: op}
		if backend == "cpu" {
			cpuOps[k] = cp
		} else if backend == "gpu" {
			gpuOps[k] = cp
		}
	}
	defer func() { debugOpHook = nil }()

	_ = m.Generate(tok.Encode("Hello"), 1)
	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()
	mgpu, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 gpu model: %v", err)
	}
	mgpu.Tok = tok
	g, err := LoadGPUModel(mgpu)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	t.Cleanup(g.Close)
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	for _, k := range topOps(cpuOps) {
		gv, ok := gpuOps[k]
		if !ok {
			t.Fatalf("missing gpu op trace for layer=%d op=%s", k.layer, k.op)
		}
		maxAbs, meanAbs := diffStats(cpuOps[k], gv)
		t.Logf("layer %d op %-8s: maxAbs=%.6g meanAbs=%.6g", k.layer, k.op, maxAbs, meanAbs)
	}
	for k := range gpuOps {
		if _, ok := cpuOps[k]; !ok {
			// GPU may trace ops absent on shared CPU K/V paths; only fail for expected ops.
			if k.op == "k" || k.op == "v" {
				if !m.Layers[k.layer].HasKV {
					continue
				}
			}
			t.Fatalf("unexpected gpu-only op trace layer=%d op=%s", k.layer, k.op)
		}
	}

	// Strongly assert that at least one problematic projection is still far off, to lock in evidence.
	k := opTraceKey{layer: 14, op: "q"}
	maxAbs, _ := diffStats(cpuOps[k], gpuOps[k])
	if maxAbs < 1e-3 || math.IsInf(maxAbs, 0) {
		t.Fatalf("expected non-trivial layer 14 q drift, got maxAbs=%g", maxAbs)
	}
}
