package config

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// ReadJSON reads a JSON file into dst and returns the raw bytes for callers
// that need to inspect alternate/nested config shapes.
func ReadJSON(path string, dst any) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, dst); err != nil {
		return nil, err
	}
	return data, nil
}

// ReadOptionalJSON reads a JSON file if it exists. The returned bool is true
// only when the file existed and decoded successfully.
func ReadOptionalJSON(path string, dst any) (bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if err := json.Unmarshal(data, dst); err != nil {
		return false, err
	}
	return true, nil
}

// ReadModelConfig reads <dir>/config.json into dst and returns the raw bytes.
func ReadModelConfig(dir string, dst any) ([]byte, error) {
	return ReadJSON(filepath.Join(dir, "config.json"), dst)
}

// ReadQuantizeConfig reads <dir>/quantize_config.json if present.
func ReadQuantizeConfig(dir string, dst any) (bool, error) {
	return ReadOptionalJSON(filepath.Join(dir, "quantize_config.json"), dst)
}
