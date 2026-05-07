package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedLayer7FinalTailSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer7 final-tail sensitivity")
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
		t.Fatalf("load gemma4 quantized model: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
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
		if step != traceStep || layer != 7 || op != "hidden_post_pli" {
			return
		}
		cp := append([]float32(nil), vec...)
		if backend == "cpu" {
			cpuOps[opTraceKey{layer, op}] = cp
		} else if backend == "gpu" {
			gpuOps[opTraceKey{layer, op}] = cp
		}
	}
	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if step != traceStep || layer != 7 {
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

	cpuHiddenPostPLI := cpuOps[opTraceKey{7, "hidden_post_pli"}]
	gpuHiddenPostPLI := gpuOps[opTraceKey{7, "hidden_post_pli"}]
	cpuHiddenFinal := cpuFinal[7]
	gpuHiddenFinal := gpuFinal[7]
	if len(cpuHiddenPostPLI) == 0 || len(gpuHiddenPostPLI) == 0 || len(cpuHiddenFinal) == 0 || len(gpuHiddenFinal) == 0 {
		t.Fatal("missing layer7 hidden_post_pli/final traces")
	}

	s := m.Layers[7].LayerScalar
	cpuRecon := append([]float32(nil), cpuHiddenPostPLI...)
	for i := range cpuRecon {
		cpuRecon[i] *= s
	}
	bf16Slice(cpuRecon)
	maxAbs, meanAbs := diffStats(cpuHiddenFinal, cpuRecon)
	t.Logf("L7 cpu bf16(layerScalar*hidden_post_pli) vs cpu final_hidden: maxAbs=%.6g meanAbs=%.6g scalar=%.6g", maxAbs, meanAbs, s)

	gpuRecon := append([]float32(nil), gpuHiddenPostPLI...)
	for i := range gpuRecon {
		gpuRecon[i] *= s
	}
	bf16Slice(gpuRecon)
	maxAbs, meanAbs = diffStats(gpuHiddenFinal, gpuRecon)
	t.Logf("L7 cpu bf16(layerScalar*gpu hidden_post_pli) vs gpu final_hidden: maxAbs=%.6g meanAbs=%.6g scalar=%.6g", maxAbs, meanAbs, s)
}
