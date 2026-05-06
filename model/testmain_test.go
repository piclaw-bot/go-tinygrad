package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestMain(m *testing.M) {
	code := m.Run()
	gpu.Shutdown()
	os.Exit(code)
}
