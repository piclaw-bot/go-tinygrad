package model

import (
	"os"
	"testing"
	"time"
)

func TestGemma4LoadTimeBreakdown(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	// Cold load
	t0 := time.Now()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tCold := time.Since(t0)
	_ = m

	// Warm load (OS page cache should be hot)
	t1 := time.Now()
	m2, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load2: %v", err)
	}
	tWarm := time.Since(t1)
	_ = m2

	t.Logf("cold load: %.2fs", tCold.Seconds())
	t.Logf("warm load: %.2fs", tWarm.Seconds())
}
