//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/backends/simd"
	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4Layer0InputNormOnModelBuffersVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 model-buffer input-norm trace")
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
		t.Fatalf("load gemma4 cpu model: %v", err)
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
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if step != traceStep || layer != 0 {
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

	cpuHidden := cpuOps[opTraceKey{0, "hidden_in"}]
	gpuHidden := gpuOps[opTraceKey{0, "hidden_in"}]
	gpuNormed := gpuOps[opTraceKey{0, "normed"}]
	if len(cpuHidden) == 0 || len(gpuHidden) == 0 || len(gpuNormed) == 0 {
		t.Fatal("missing layer0 traces")
	}

	check := func(label string, hidden []float32) {
		copy(g.hidden.Data(), hidden)
		g.hidden.MarkDirty()
		gpu.DevRMSNorm(g.normed, g.hidden, g.Layers[0].InputNorm, float32(m.Config.RMSNormEps))
		gpu.DevToBF16(g.normed, len(hidden))
		gpu.Sync()
		got := append([]float32(nil), g.normed.Data()[:len(hidden)]...)

		want := append([]float32(nil), hidden...)
		simd.RMSNormBF16(want, m.Layers[0].InputNorm.Data(), float32(m.Config.RMSNormEps))

		maxAbs, meanAbs := diffStats(want, got)
		traceMax, traceMean := diffStats(gpuNormed, got)
		t.Logf("modelbuf inputnorm %-8s vs cpu: maxAbs=%.6g meanAbs=%.6g", label, maxAbs, meanAbs)
		t.Logf("modelbuf inputnorm %-8s vs traced gpuNormed: maxAbs=%.6g meanAbs=%.6g", label, traceMax, traceMean)
	}

	check("cpuHidden", cpuHidden)
	check("gpuHidden", gpuHidden)
}
