package model

import (
	"fmt"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

type stepLayerKey struct {
	step  int
	layer int
}

type sensitivityTrace struct {
	layers map[int][]float32
	logits []float32
}

func captureCPUQuantizedTrace(t *testing.T, dir string, tok *Tokenizer, prompt string, targets map[int]bool, overrideLayer int, override map[int][]float32) sensitivityTrace {
	t.Helper()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load quantized cpu model: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, prompt)
	traceStep := len(wrapped) - 1
	tr := sensitivityTrace{layers: map[int][]float32{}}

	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if backend != "cpu" || step != traceStep || !targets[layer] {
			return
		}
		tr.layers[layer] = append([]float32(nil), hidden...)
	}
	debugLogitsHook = func(backend string, step int, hidden, logits []float32) {
		if backend != "cpu" || step != traceStep {
			return
		}
		tr.logits = append([]float32(nil), logits...)
	}
	if override != nil {
		debugCPUHiddenInOverrideHook = func(step, layer int, hidden []float32) {
			if layer != overrideLayer {
				return
			}
			if repl, ok := override[step]; ok && len(repl) == len(hidden) {
				copy(hidden, repl)
			}
		}
	}
	defer func() {
		debugLayerHook = nil
		debugLogitsHook = nil
		debugCPUHiddenInOverrideHook = nil
	}()

	_ = m.Generate(tok.Encode(prompt), 1)
	return tr
}

func captureGPUQuantizedTraceWithSteps(t *testing.T, dir string, tok *Tokenizer, prompt string, targets map[int]bool, captureStepLayers map[int]bool) (sensitivityTrace, map[stepLayerKey][]float32) {
	t.Helper()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load quantized gpu model: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, prompt)
	traceStep := len(wrapped) - 1
	tr := sensitivityTrace{layers: map[int][]float32{}}
	stepLayers := map[stepLayerKey][]float32{}

	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok

	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if backend != "gpu" {
			return
		}
		if step == traceStep && targets[layer] {
			tr.layers[layer] = append([]float32(nil), hidden...)
		}
		if captureStepLayers[layer] {
			stepLayers[stepLayerKey{step: step, layer: layer}] = append([]float32(nil), hidden...)
		}
	}
	debugLogitsHook = func(backend string, step int, hidden, logits []float32) {
		if backend != "gpu" || step != traceStep {
			return
		}
		tr.logits = append([]float32(nil), logits...)
	}
	defer func() {
		debugLayerHook = nil
		debugLogitsHook = nil
	}()

	_ = g.Generate(tok.Encode(prompt), 1)
	return tr, stepLayers
}

func logTraceDiff(t *testing.T, label string, want, got sensitivityTrace, tok *Tokenizer) {
	t.Helper()
	for _, l := range []int{14, 15, 34} {
		w, ok1 := want.layers[l]
		g, ok2 := got.layers[l]
		if !ok1 || !ok2 {
			t.Fatalf("%s missing layer trace %d (want=%v got=%v)", label, l, ok1, ok2)
		}
		maxAbs, meanAbs := diffStats(w, g)
		t.Logf("%s layer %d: maxAbs=%.6g meanAbs=%.6g", label, l, maxAbs, meanAbs)
	}
	if len(want.logits) == 0 || len(got.logits) == 0 {
		t.Fatalf("%s missing final logits", label)
	}
	maxAbs, meanAbs := diffStats(want.logits, got.logits)
	wt := topLogits(want.logits, 5)
	gt := topLogits(got.logits, 5)
	t.Logf("%s final logits: maxAbs=%.6g meanAbs=%.6g", label, maxAbs, meanAbs)
	t.Logf("%s want top5 ids=%v toks=%v", label, wt, tokenNames(tok, wt))
	t.Logf("%s got  top5 ids=%v toks=%v", label, gt, tokenNames(tok, gt))
}

func TestGemma4QuantizedForwardSensitivityFromLayer0And1(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized forward sensitivity")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}
	t.Cleanup(gpu.Shutdown)

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load quantized model: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	prompt := "Hello"
	targets := map[int]bool{14: true, 15: true, 34: true}
	captureStepLayers := map[int]bool{0: true, 1: true}

	gpuTrace, gpuStepLayers := captureGPUQuantizedTraceWithSteps(t, dir, tok, prompt, targets, captureStepLayers)
	cpuTrace := captureCPUQuantizedTrace(t, dir, tok, prompt, targets, -1, nil)
	logTraceDiff(t, "cpuq_vs_gpu_baseline", gpuTrace, cpuTrace, tok)

	overrideFromL0 := map[int][]float32{}
	overrideFromL1 := map[int][]float32{}
	for k, v := range gpuStepLayers {
		switch k.layer {
		case 0:
			overrideFromL0[k.step] = v
		case 1:
			overrideFromL1[k.step] = v
		}
	}
	if len(overrideFromL0) == 0 || len(overrideFromL1) == 0 {
		t.Fatalf("missing captured gpu step layers: l0=%d l1=%d", len(overrideFromL0), len(overrideFromL1))
	}

	cpuFromL0 := captureCPUQuantizedTrace(t, dir, tok, prompt, targets, 1, overrideFromL0)
	logTraceDiff(t, "cpuq_replayed_from_gpu_l0_at_l1", gpuTrace, cpuFromL0, tok)

	cpuFromL1 := captureCPUQuantizedTrace(t, dir, tok, prompt, targets, 2, overrideFromL1)
	logTraceDiff(t, "cpuq_replayed_from_gpu_l1_at_l2", gpuTrace, cpuFromL1, tok)

	fmt.Printf("[gemma4-forward-sensitivity] captured steps: layer0=%d layer1=%d\n", len(overrideFromL0), len(overrideFromL1))
}
