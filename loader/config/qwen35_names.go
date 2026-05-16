package config

import "strings"

const Qwen35NestedTextPrefix = "model.language_model."

func NormalizeQwen35TensorName(name string) string {
	name = strings.TrimPrefix(name, Qwen35NestedTextPrefix)
	name = strings.TrimPrefix(name, "language_model.")
	return name
}

func Qwen35TensorNameCandidates(name string) []string {
	norm := NormalizeQwen35TensorName(name)
	if strings.HasPrefix(norm, "mtp.") {
		return []string{norm}
	}
	return []string{norm, Qwen35NestedTextPrefix + norm, "language_model." + norm}
}

func IsQwen35MainLayerTensorName(name string) bool {
	norm := NormalizeQwen35TensorName(name)
	return strings.HasPrefix(norm, "model.layers.")
}
