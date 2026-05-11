//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

type frontStepOpKey struct {
	step int
	op   string
}

func captureCPUQuantizedFrontStack(t *testing.T, dir string, tok *tokenizer.Tokenizer, prompt string, targets map[int]bool, overrideLayer int, override map[int][]float32, pliOverride map[int][]float32, mlpOverrideLayer int, mlpOverride map[int][]float32) (sensitivityTrace, map[frontStepOpKey][]float32, map[int][]float32) {
	t.Helper()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load quantized cpu model: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, prompt)
	traceStep := len(wrapped) - 1
	tr := sensitivityTrace{layers: map[int][]float32{}}
	opSteps := map[frontStepOpKey][]float32{}
	layer0OutByStep := map[int][]float32{}

	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if backend != "cpu" {
			return
		}
		if layer == 0 {
			layer0OutByStep[step] = append([]float32(nil), hidden...)
		}
		if step == traceStep && targets[layer] {
			tr.layers[layer] = append([]float32(nil), hidden...)
		}
	}
	debugLogitsHook = func(backend string, step int, hidden, logits []float32) {
		if backend != "cpu" || step != traceStep {
			return
		}
		tr.logits = append([]float32(nil), logits...)
	}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || layer != 0 {
			return
		}
		switch op {
		case "embed_scaled", "pli0_input", "hidden_in", "mlp_input":
			opSteps[frontStepOpKey{step: step, op: op}] = append([]float32(nil), vec...)
		}
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
	if pliOverride != nil {
		debugCPUPerLayerInputsOverrideHook = func(step int, perLayerInputs [][]float32) {
			if repl, ok := pliOverride[step]; ok && len(perLayerInputs) > 0 && len(perLayerInputs[0]) == len(repl) {
				copy(perLayerInputs[0], repl)
			}
		}
	}
	if mlpOverride != nil {
		debugCPUMLPInputOverrideHook = func(step, layer int, mlpInput []float32) {
			if layer != mlpOverrideLayer {
				return
			}
			if repl, ok := mlpOverride[step]; ok && len(repl) == len(mlpInput) {
				copy(mlpInput, repl)
			}
		}
	}
	defer func() {
		debugLayerHook = nil
		debugLogitsHook = nil
		debugOpHook = nil
		debugCPUHiddenInOverrideHook = nil
		debugCPUPerLayerInputsOverrideHook = nil
		debugCPUMLPInputOverrideHook = nil
	}()

	_ = m.Generate(tok.Encode(prompt), 1)
	return tr, opSteps, layer0OutByStep
}

func captureGPUQuantizedFrontStack(t *testing.T, dir string, tok *tokenizer.Tokenizer, prompt string, targets map[int]bool) (sensitivityTrace, map[frontStepOpKey][]float32, map[int][]float32) {
	t.Helper()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load quantized gpu model: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, prompt)
	traceStep := len(wrapped) - 1
	tr := sensitivityTrace{layers: map[int][]float32{}}
	opSteps := map[frontStepOpKey][]float32{}
	layer0OutByStep := map[int][]float32{}

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
		if layer == 0 {
			layer0OutByStep[step] = append([]float32(nil), hidden...)
		}
		if step == traceStep && targets[layer] {
			tr.layers[layer] = append([]float32(nil), hidden...)
		}
	}
	debugLogitsHook = func(backend string, step int, hidden, logits []float32) {
		if backend != "gpu" || step != traceStep {
			return
		}
		tr.logits = append([]float32(nil), logits...)
	}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "gpu" || layer != 0 {
			return
		}
		switch op {
		case "embed_scaled", "pli0_input", "hidden_in", "mlp_input":
			opSteps[frontStepOpKey{step: step, op: op}] = append([]float32(nil), vec...)
		}
	}
	defer func() {
		debugLayerHook = nil
		debugLogitsHook = nil
		debugOpHook = nil
	}()

	_ = g.Generate(tok.Encode(prompt), 1)
	return tr, opSteps, layer0OutByStep
}

func logWorstFrontStepDiff(t *testing.T, label string, cpuSteps, gpuSteps map[frontStepOpKey][]float32, op string) {
	t.Helper()
	worstStep := -1
	worstMax := -1.0
	worstMean := 0.0
	for k, g := range gpuSteps {
		if k.op != op {
			continue
		}
		c, ok := cpuSteps[k]
		if !ok {
			continue
		}
		maxAbs, meanAbs := diffStats(c, g)
		if maxAbs > worstMax {
			worstMax = maxAbs
			worstMean = meanAbs
			worstStep = k.step
		}
	}
	t.Logf("%s worst %s step=%d maxAbs=%.6g meanAbs=%.6g", label, op, worstStep, worstMax, worstMean)
}

func TestGemma4QuantizedFrontStackSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized front-stack sensitivity")
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
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	prompt := "Hello"
	targets := map[int]bool{14: true, 15: true, 34: true}

	gpuTrace, gpuOps, gpuLayer0Out := captureGPUQuantizedFrontStack(t, dir, tok, prompt, targets)
	cpuTrace, cpuOps, _ := captureCPUQuantizedFrontStack(t, dir, tok, prompt, targets, -1, nil, nil, -1, nil)
	logTraceDiff(t, "cpuq_vs_gpu_baseline", gpuTrace, cpuTrace, tok)
	logWorstFrontStepDiff(t, "frontstack", cpuOps, gpuOps, "embed_scaled")
	logWorstFrontStepDiff(t, "frontstack", cpuOps, gpuOps, "pli0_input")
	logWorstFrontStepDiff(t, "frontstack", cpuOps, gpuOps, "hidden_in")
	logWorstFrontStepDiff(t, "frontstack", cpuOps, gpuOps, "mlp_input")

	overrideFromHiddenIn := map[int][]float32{}
	for k, v := range gpuOps {
		if k.op == "hidden_in" {
			overrideFromHiddenIn[k.step] = v
		}
	}
	if len(overrideFromHiddenIn) == 0 || len(gpuLayer0Out) == 0 {
		t.Fatalf("missing captured front-stack state: hidden_in=%d layer0out=%d", len(overrideFromHiddenIn), len(gpuLayer0Out))
	}

	cpuFromHiddenIn, _, _ := captureCPUQuantizedFrontStack(t, dir, tok, prompt, targets, 0, overrideFromHiddenIn, nil, -1, nil)
	logTraceDiff(t, "cpuq_replayed_from_gpu_l0_hidden_in_at_l0", gpuTrace, cpuFromHiddenIn, tok)

	pliOverride := map[int][]float32{}
	for k, v := range gpuOps {
		if k.op == "pli0_input" {
			pliOverride[k.step] = v
		}
	}
	if len(pliOverride) == 0 {
		t.Fatal("missing captured gpu pli0_input states")
	}
	cpuFromPLI0, _, _ := captureCPUQuantizedFrontStack(t, dir, tok, prompt, targets, -1, nil, pliOverride, -1, nil)
	logTraceDiff(t, "cpuq_replayed_from_gpu_pli0_input", gpuTrace, cpuFromPLI0, tok)

	mlpOverride := map[int][]float32{}
	for k, v := range gpuOps {
		if k.op == "mlp_input" {
			mlpOverride[k.step] = v
		}
	}
	if len(mlpOverride) == 0 {
		t.Fatal("missing captured gpu layer0 mlp_input states")
	}
	cpuFromMLPInput, _, _ := captureCPUQuantizedFrontStack(t, dir, tok, prompt, targets, -1, nil, nil, 0, mlpOverride)
	logTraceDiff(t, "cpuq_replayed_from_gpu_l0_mlp_input", gpuTrace, cpuFromMLPInput, tok)

	cpuFromLayer0Out, _, _ := captureCPUQuantizedFrontStack(t, dir, tok, prompt, targets, 1, gpuLayer0Out, nil, -1, nil)
	logTraceDiff(t, "cpuq_replayed_from_gpu_l0_output_at_l1", gpuTrace, cpuFromLayer0Out, tok)
}
