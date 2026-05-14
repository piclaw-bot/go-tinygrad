package config

import (
	"encoding/json"
	"strings"
)

// QuantizationMetadata is a small normalized view over the quantization
// declarations used by Hugging Face config.json files. It intentionally keeps
// the original strings needed for diagnostics while exposing only the fields
// currently needed by loaders.
type QuantizationMetadata struct {
	Algo           string
	Method         string
	Format         string
	Bits           int
	GroupSize      int
	Symmetric      bool
	HasConfig      bool
	UnsupportedFP4 bool
}

// ParseQuantizationMetadata extracts common quantization metadata from a raw
// config.json payload. It supports the MLX shape used by local checkpoints, and
// Hugging Face quantization_config declarations used by ModelOpt and
// compressed-tensors checkpoints.
func ParseQuantizationMetadata(data []byte) (QuantizationMetadata, error) {
	var raw struct {
		Quantization struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		} `json:"quantization"`
		QuantizationConfig struct {
			QuantAlgo    string `json:"quant_algo"`
			QuantMethod  string `json:"quant_method"`
			Format       string `json:"format"`
			Bits         int    `json:"bits"`
			GroupSize    int    `json:"group_size"`
			Sym          bool   `json:"sym"`
			ConfigGroups map[string]struct {
				Format  string `json:"format"`
				Weights struct {
					NumBits   int    `json:"num_bits"`
					Type      string `json:"type"`
					Format    string `json:"format"`
					GroupSize int    `json:"group_size"`
					Symmetric bool   `json:"symmetric"`
				} `json:"weights"`
			} `json:"config_groups"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return QuantizationMetadata{}, err
	}

	md := QuantizationMetadata{}
	if raw.Quantization.Bits > 0 {
		md.HasConfig = true
		md.Method = "mlx"
		md.Bits = raw.Quantization.Bits
		md.GroupSize = raw.Quantization.GroupSize
	}

	fp4FormatSeen := false
	qc := raw.QuantizationConfig
	if qc.QuantAlgo != "" || qc.QuantMethod != "" || qc.Format != "" || qc.Bits > 0 || len(qc.ConfigGroups) > 0 {
		md.HasConfig = true
		md.Algo = qc.QuantAlgo
		md.Method = qc.QuantMethod
		md.Format = qc.Format
		if qc.Bits > 0 {
			md.Bits = qc.Bits
		}
		if qc.GroupSize > 0 {
			md.GroupSize = qc.GroupSize
		}
		md.Symmetric = qc.Sym
		for _, group := range qc.ConfigGroups {
			if isFP4FormatString(group.Format) || isFP4FormatString(group.Weights.Format) || (group.Weights.NumBits == 4 && isFloatFormatString(group.Weights.Type)) {
				fp4FormatSeen = true
			}
			if group.Weights.NumBits > 0 && md.Bits == 0 {
				md.Bits = group.Weights.NumBits
			}
			if group.Weights.GroupSize > 0 && md.GroupSize == 0 {
				md.GroupSize = group.Weights.GroupSize
			}
			if group.Weights.Symmetric {
				md.Symmetric = true
			}
			if md.Format == "" {
				if group.Format != "" {
					md.Format = group.Format
				} else if group.Weights.Format != "" {
					md.Format = group.Weights.Format
				} else {
					md.Format = group.Weights.Type
				}
			}
		}
	}

	algo := strings.ToLower(md.Algo)
	method := strings.ToLower(md.Method)
	format := strings.ToLower(md.Format)
	md.UnsupportedFP4 = strings.Contains(algo, "nvfp4") || strings.Contains(algo, "fp4") || isFP4FormatString(format) || strings.Contains(method, "modelopt") || (md.Bits == 4 && isFloatFormatString(format)) || fp4FormatSeen
	return md, nil
}

func isFP4FormatString(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "nvfp4") || strings.Contains(s, "fp4")
}

func isFloatFormatString(s string) bool {
	return strings.Contains(strings.ToLower(s), "float")
}
