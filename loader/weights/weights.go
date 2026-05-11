package weights

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Source is the common tensor lookup surface used by model loaders.
type Source interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetInt32(name string) ([]int32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
	Close() error
}

// EagerSource is implemented by sources that can pre-fault mapped backing
// storage before model load touches individual tensors.
type EagerSource interface {
	EagerLoad() (int64, error)
}

// OpenSafetensors opens either a sharded safetensors index or a single
// model.safetensors file from dir.
func OpenSafetensors(dir string) (Source, error) {
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(indexPath); err == nil {
		sf, err := safetensors.OpenSharded(indexPath)
		if err != nil {
			return nil, fmt.Errorf("open sharded: %w", err)
		}
		return sf, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("stat %s: %w", indexPath, err)
	}

	sf, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("open single: %w", err)
	}
	return sf, nil
}
