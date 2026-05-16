package model

import (
	"fmt"
	"os"
	"path/filepath"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

type Qwen35NativeMTPBundle struct {
	Meta loaderconfig.QwenNativeMTPMetadata
	Base *Qwen35BaseModel
	MTP  *QwenNativeMTPHead
}

func (b *Qwen35NativeMTPBundle) ValidateNativeMTPReady() error {
	if b == nil {
		return fmt.Errorf("nil Qwen3.5 native MTP bundle")
	}
	if !b.Meta.HasNativeMTP {
		return fmt.Errorf("Qwen3.5 native MTP is not enabled in metadata")
	}
	if b.MTP == nil {
		return fmt.Errorf("Qwen3.5 native MTP head is not loaded")
	}
	return ValidateQwenNativeMTPHead(b.MTP, b.Meta)
}

func (b *Qwen35NativeMTPBundle) NewForwardState() (Qwen35BaseForwardState, error) {
	if b == nil {
		return Qwen35BaseForwardState{}, fmt.Errorf("nil Qwen3.5 native MTP bundle")
	}
	return NewQwen35BaseForwardState(b.Base, b.Meta)
}

func (b *Qwen35NativeMTPBundle) ForwardBaseSequence(inputs [][]float32, state Qwen35BaseForwardState, ropeFreqs []float32, eps float32) ([][]float32, Qwen35BaseForwardState, error) {
	if b == nil || b.Base == nil {
		return nil, state, fmt.Errorf("nil Qwen3.5 base model in bundle")
	}
	return b.Base.ForwardSequence(inputs, state, ropeFreqs, eps, b.Meta)
}

func LoadQwen35NativeMTPBundleFromDir(dir string) (*Qwen35NativeMTPBundle, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("read Qwen3.5 config: %w", err)
	}
	meta, err := loaderconfig.ParseQwenNativeMTPMetadata(data)
	if err != nil {
		return nil, fmt.Errorf("parse Qwen3.5 config: %w", err)
	}
	base, err := LoadQwen35BaseModelLayersFromSafetensorsDir(dir, meta)
	if err != nil {
		return nil, fmt.Errorf("load Qwen3.5 base layers: %w", err)
	}
	bundle := &Qwen35NativeMTPBundle{Meta: meta, Base: base}
	if meta.HasNativeMTP {
		mtp, err := LoadQwenNativeMTPHeadFromSafetensorsDir(dir, meta)
		if err != nil {
			return nil, fmt.Errorf("load Qwen native MTP head: %w", err)
		}
		bundle.MTP = mtp
	}
	return bundle, nil
}
