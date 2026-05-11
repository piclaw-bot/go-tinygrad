package weights

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenSafetensorsReportsShardedOpenError(t *testing.T) {
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.Mkdir(indexPath, 0o755); err != nil {
		t.Fatal(err)
	}
	_, err := OpenSafetensors(dir)
	if err == nil {
		t.Fatal("OpenSafetensors accepted directory index path")
	}
	if !strings.Contains(err.Error(), "open sharded") {
		t.Fatalf("err=%v, want open sharded error", err)
	}
}

func TestOpenSafetensorsMissingSingle(t *testing.T) {
	_, err := OpenSafetensors(t.TempDir())
	if err == nil {
		t.Fatal("OpenSafetensors accepted missing single file")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("err=%v, want os.ErrNotExist", err)
	}
}
