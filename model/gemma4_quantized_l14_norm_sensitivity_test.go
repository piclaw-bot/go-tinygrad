package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedLayer14NormSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer14 norm sensitivity")
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
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	cpuOps := map[opTraceKey][]float32{}
	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if step != traceStep || layer != 14 {
			return
		}
		if op != "hidden_in" && op != "normed" {
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

	cpuHidden := cpuOps[opTraceKey{14, "hidden_in"}]
	cpuNormed := cpuOps[opTraceKey{14, "normed"}]
	gpuHidden := gpuOps[opTraceKey{14, "hidden_in"}]
	gpuNormed := gpuOps[opTraceKey{14, "normed"}]
	if len(cpuHidden) == 0 || len(cpuNormed) == 0 || len(gpuHidden) == 0 || len(gpuNormed) == 0 {
		t.Fatal("missing layer14 hidden_in/normed traces")
	}

	cpuRecon := append([]float32(nil), gpuHidden...)
	rmsNormInPlace(cpuRecon, m.Layers[14].InputNorm.Data(), float32(m.Config.RMSNormEps))
	maxAbs, meanAbs := diffStats(gpuNormed, cpuRecon)
	t.Logf("L14 cpu(norm(gpuHidden)) vs gpu normed: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	maxAbs, meanAbs = diffStats(cpuNormed, cpuRecon)
	t.Logf("L14 cpu(norm(gpuHidden)) vs cpu normed: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	hidBuf := gpu.NewDevBufFrom(append([]float32(nil), gpuHidden...))
	outBuf := gpu.NewDevBuf(len(gpuHidden))
	defer hidBuf.Free()
	defer outBuf.Free()
	if err := hidBuf.ToGPU(); err != nil {
		t.Fatalf("hidBuf.ToGPU: %v", err)
	}
	if err := outBuf.ToGPU(); err != nil {
		t.Fatalf("outBuf.ToGPU: %v", err)
	}
	gpu.DevRMSNorm(outBuf, hidBuf, g.Layers[14].InputNorm, float32(m.Config.RMSNormEps))
	gpu.Sync()
	gpuRecon := append([]float32(nil), outBuf.Data()[:len(gpuNormed)]...)
	maxAbs, meanAbs = diffStats(gpuNormed, gpuRecon)
	t.Logf("L14 fresh gpu(norm(gpuHidden)) vs gpu normed: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
