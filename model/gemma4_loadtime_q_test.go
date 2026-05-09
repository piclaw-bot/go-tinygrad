package model

import (
	"os"
	"testing"
	"time"
)

func TestGemma4LoadTimeQuantized(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()

	t0 := time.Now()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tLoad := time.Since(t0)
	_ = m

	t.Logf("quantized load: %.2fs", tLoad.Seconds())
}
