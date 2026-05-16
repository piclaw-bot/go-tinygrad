package config

import "testing"

func TestNormalizeQwen35TensorName(t *testing.T) {
	cases := map[string]string{
		"model.layers.0.self_attn.q_proj.weight":                      "model.layers.0.self_attn.q_proj.weight",
		"language_model.model.layers.0.self_attn.q_proj.weight":       "model.layers.0.self_attn.q_proj.weight",
		"model.language_model.model.layers.0.self_attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
		"model.language_model.mtp.layers.0.self_attn.q_proj.weight":   "mtp.layers.0.self_attn.q_proj.weight",
	}
	for in, want := range cases {
		if got := NormalizeQwen35TensorName(in); got != want {
			t.Fatalf("NormalizeQwen35TensorName(%q)=%q want %q", in, got, want)
		}
	}
}

func TestQwen35TensorNameCandidates(t *testing.T) {
	got := Qwen35TensorNameCandidates("model.layers.0.self_attn.q_proj.weight")
	want := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.language_model.model.layers.0.self_attn.q_proj.weight",
		"model.language_model.layers.0.self_attn.q_proj.weight",
		"language_model.model.layers.0.self_attn.q_proj.weight",
	}
	if len(got) != len(want) {
		t.Fatalf("candidates=%v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("candidate %d=%q want %q", i, got[i], want[i])
		}
	}
	mtp := Qwen35TensorNameCandidates("model.language_model.mtp.fc.weight")
	if len(mtp) != 1 || mtp[0] != "mtp.fc.weight" {
		t.Fatalf("mtp candidates=%v", mtp)
	}
}

func TestQwen35TensorNameCandidatesIncludeReferenceAliases(t *testing.T) {
	got := Qwen35TensorNameCandidates("model.layers.0.linear_attn.in_proj_qkvz.weight")
	want := "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
	for _, candidate := range got {
		if candidate == want {
			return
		}
	}
	t.Fatalf("missing alias %q in %v", want, got)
}

func TestIsQwen35MainLayerTensorName(t *testing.T) {
	for _, name := range []string{
		"model.layers.0.self_attn.q_proj.weight",
		"language_model.model.layers.0.self_attn.q_proj.weight",
		"model.language_model.model.layers.0.linear_attn.in_proj_qkvz.weight",
	} {
		if !IsQwen35MainLayerTensorName(name) {
			t.Fatalf("%q not recognized", name)
		}
	}
	if IsQwen35MainLayerTensorName("mtp.layers.0.self_attn.q_proj.weight") {
		t.Fatal("MTP layer recognized as main layer")
	}
}
