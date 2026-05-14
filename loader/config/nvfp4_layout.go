package config

import "strings"

// NVFP4TensorRole classifies an observed tensor prefix in ModelOpt/NVFP4
// checkpoints. The prefix is the tensor name without .weight/.weight_scale
// suffixes, e.g. model.layers.0.self_attn.q_proj.
type NVFP4TensorRole string

const (
	NVFP4RoleUnknown       NVFP4TensorRole = "unknown"
	NVFP4RoleAttentionQ    NVFP4TensorRole = "attention_q"
	NVFP4RoleAttentionK    NVFP4TensorRole = "attention_k"
	NVFP4RoleAttentionV    NVFP4TensorRole = "attention_v"
	NVFP4RoleAttentionO    NVFP4TensorRole = "attention_o"
	NVFP4RoleMLPGate       NVFP4TensorRole = "mlp_gate"
	NVFP4RoleMLPUp         NVFP4TensorRole = "mlp_up"
	NVFP4RoleMLPDown       NVFP4TensorRole = "mlp_down"
	NVFP4RoleRouter        NVFP4TensorRole = "router"
	NVFP4RoleMoEExpertGate NVFP4TensorRole = "moe_expert_gate"
	NVFP4RoleMoEExpertUp   NVFP4TensorRole = "moe_expert_up"
	NVFP4RoleMoEExpertDown NVFP4TensorRole = "moe_expert_down"
)

// ClassifyNVFP4TensorPrefix maps Qwen/Gemma linear tensor prefixes to stable
// roles used by loader integration. It intentionally handles dense and MoE
// expert names without binding to a specific top-level model prefix.
func ClassifyNVFP4TensorPrefix(prefix string) NVFP4TensorRole {
	if isNVFP4MoEExpertPrefix(prefix) {
		switch {
		case strings.HasSuffix(prefix, ".gate_proj"):
			return NVFP4RoleMoEExpertGate
		case strings.HasSuffix(prefix, ".up_proj"):
			return NVFP4RoleMoEExpertUp
		case strings.HasSuffix(prefix, ".down_proj"):
			return NVFP4RoleMoEExpertDown
		}
	}
	switch {
	case strings.HasSuffix(prefix, ".self_attn.q_proj"):
		return NVFP4RoleAttentionQ
	case strings.HasSuffix(prefix, ".self_attn.k_proj"):
		return NVFP4RoleAttentionK
	case strings.HasSuffix(prefix, ".self_attn.v_proj"):
		return NVFP4RoleAttentionV
	case strings.HasSuffix(prefix, ".self_attn.o_proj"):
		return NVFP4RoleAttentionO
	case strings.HasSuffix(prefix, ".mlp.gate_proj"):
		return NVFP4RoleMLPGate
	case strings.HasSuffix(prefix, ".mlp.up_proj"):
		return NVFP4RoleMLPUp
	case strings.HasSuffix(prefix, ".mlp.down_proj"):
		return NVFP4RoleMLPDown
	case strings.HasSuffix(prefix, ".mlp.gate"):
		return NVFP4RoleRouter
	default:
		return NVFP4RoleUnknown
	}
}

func isNVFP4MoEExpertPrefix(prefix string) bool {
	return strings.Contains(prefix, ".mlp.experts.") || (strings.Contains(prefix, ".layers.") && strings.Contains(prefix, ".experts."))
}

// NVFP4CompanionNames returns the ModelOpt companion tensor names for a
// quantized linear tensor prefix.
func NVFP4CompanionNames(prefix string) (weight, weightScale, weightScale2, inputScale string) {
	return prefix + ".weight", prefix + ".weight_scale", prefix + ".weight_scale_2", prefix + ".input_scale"
}
