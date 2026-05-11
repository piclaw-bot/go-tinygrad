package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedLayer6TransitionTrace(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer6 transition trace")
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
		t.Fatalf("load gemma4 quantized cpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	cpuOps := map[opTraceKey][]float32{}
	gpuOps := map[opTraceKey][]float32{}
	cpuFinal := map[int][]float32{}
	gpuFinal := map[int][]float32{}

	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if step != traceStep {
			return
		}
		if layer != 6 && layer != 7 {
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
	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if step != traceStep {
			return
		}
		if layer != 6 && layer != 7 {
			return
		}
		cp := append([]float32(nil), hidden...)
		if backend == "cpu" {
			cpuFinal[layer] = cp
		} else if backend == "gpu" {
			gpuFinal[layer] = cp
		}
	}
	defer func() {
		debugOpHook = nil
		debugLayerHook = nil
	}()

	_ = m.Generate(tok.Encode("Hello"), 1)

	mgpu, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized gpu model: %v", err)
	}
	mgpu.Tok = tok
	g, err := LoadGPUModel(mgpu)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	checks := []struct {
		layer int
		op    string
	}{
		{6, "down"},
		{6, "down_postffn"},
		{6, "hidden_post_ffn"},
		{6, "hidden_post_pli"},
		{7, "hidden_in"},
	}
	for _, chk := range checks {
		k := opTraceKey{layer: chk.layer, op: chk.op}
		c, ok1 := cpuOps[k]
		g2, ok2 := gpuOps[k]
		if !ok1 || !ok2 {
			t.Fatalf("missing op trace L%d %s (cpu=%v gpu=%v)", chk.layer, chk.op, ok1, ok2)
		}
		maxAbs, meanAbs := diffStats(c, g2)
		t.Logf("L%d %-15s maxAbs=%.6g meanAbs=%.6g", chk.layer, chk.op, maxAbs, meanAbs)
	}

	for _, layer := range []int{6, 7} {
		c, ok1 := cpuFinal[layer]
		g2, ok2 := gpuFinal[layer]
		if !ok1 || !ok2 {
			t.Fatalf("missing final hidden trace L%d (cpu=%v gpu=%v)", layer, ok1, ok2)
		}
		maxAbs, meanAbs := diffStats(c, g2)
		t.Logf("L%d final_hidden     maxAbs=%.6g meanAbs=%.6g", layer, maxAbs, meanAbs)
	}

	maxAbs, meanAbs := diffStats(cpuFinal[6], cpuOps[opTraceKey{7, "hidden_in"}])
	t.Logf("CPU L6 final_hidden vs L7 hidden_in: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	maxAbs, meanAbs = diffStats(gpuFinal[6], gpuOps[opTraceKey{7, "hidden_in"}])
	t.Logf("GPU L6 final_hidden vs L7 hidden_in: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
