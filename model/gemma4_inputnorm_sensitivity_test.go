package model

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/backends/simd"
	"github.com/rcarmo/go-pherence/gpu"
)

func rmsOf(x []float32) float64 {
	if len(x) == 0 {
		return 0
	}
	var ss float64
	for _, v := range x {
		fv := float64(v)
		ss += fv * fv
	}
	return math.Sqrt(ss / float64(len(x)))
}

func TestGemma4Layer0InputNormSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 input-norm sensitivity trace")
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
	cpuNormed := cpuOps[opTraceKey{0, "normed"}]
	gpuNormed := gpuOps[opTraceKey{0, "normed"}]
	if len(cpuHidden) == 0 || len(gpuHidden) == 0 || len(cpuNormed) == 0 || len(gpuNormed) == 0 {
		t.Fatal("missing hidden/normed traces for layer 0")
	}

	hMax, hMean := diffStats(cpuHidden, gpuHidden)
	nMax, nMean := diffStats(cpuNormed, gpuNormed)
	t.Logf("layer0 hidden_in diff: maxAbs=%.6g meanAbs=%.6g cpuRMS=%.6g gpuRMS=%.6g", hMax, hMean, rmsOf(cpuHidden), rmsOf(gpuHidden))
	t.Logf("layer0 normed diff  : maxAbs=%.6g meanAbs=%.6g cpuRMS=%.6g gpuRMS=%.6g", nMax, nMean, rmsOf(cpuNormed), rmsOf(gpuNormed))

	cpuOnGPUHidden := append([]float32(nil), gpuHidden...)
	simd.RMSNormBF16(cpuOnGPUHidden, m.Layers[0].InputNorm.Data(), float32(m.Config.RMSNormEps))
	reconMax, reconMean := diffStats(cpuOnGPUHidden, gpuNormed)
	cpuVsReconMax, cpuVsReconMean := diffStats(cpuNormed, cpuOnGPUHidden)
	t.Logf("layer0 cpu(inputnorm(gpuHidden)) vs gpuNormed: maxAbs=%.6g meanAbs=%.6g", reconMax, reconMean)
	t.Logf("layer0 cpuNormed vs cpu(inputnorm(gpuHidden)): maxAbs=%.6g meanAbs=%.6g", cpuVsReconMax, cpuVsReconMean)
}
