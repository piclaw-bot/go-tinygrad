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
	base := []string{norm}
	for _, alias := range qwen35CanonicalTensorAliases(norm) {
		base = append(base, alias)
	}
	out := make([]string, 0, len(base)*4)
	seen := map[string]bool{}
	for _, n := range base {
		for _, candidate := range []string{n, Qwen35NestedTextPrefix + n, "model.language_model." + strings.TrimPrefix(n, "model."), "language_model." + n} {
			if !seen[candidate] {
				seen[candidate] = true
				out = append(out, candidate)
			}
		}
	}
	return out
}

func qwen35CanonicalTensorAliases(name string) []string {
	repls := []struct{ old, new string }{
		{".linear_attn.in_proj_qkvz.weight", ".linear_attn.in_proj_qkv.weight"},
		{".linear_attn.in_proj_gate.weight", ".linear_attn.in_proj_z.weight"},
		{".linear_attn.in_proj_ba.weight", ".linear_attn.in_proj_b.weight"},
		{".linear_attn.A", ".linear_attn.A_log"},
	}
	var out []string
	for _, r := range repls {
		if strings.Contains(name, r.old) {
			out = append(out, strings.Replace(name, r.old, r.new, 1))
		}
	}
	return out
}

func IsQwen35MainLayerTensorName(name string) bool {
	norm := NormalizeQwen35TensorName(name)
	return strings.HasPrefix(norm, "model.layers.")
}
