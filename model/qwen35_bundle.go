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
