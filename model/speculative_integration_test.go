package model

import (
	"os"
	"path/filepath"
	"testing"
)

func localModelPath(name string) string {
	candidates := []string{
		filepath.Join("models", name),
		filepath.Join("..", "models", name),
		filepath.Join("..", "..", "models", name),
	}
	for _, p := range candidates {
		if _, err := os.Stat(filepath.Join(p, "config.json")); err == nil {
			return p
		}
	}
	return ""
}

func TestSpeculativeMatchesNormalSmallLocalModels(t *testing.T) {
	if os.Getenv("GO_PHERENCE_RUN_LOCAL_MODEL_TESTS") != "1" {
		t.Skip("set GO_PHERENCE_RUN_LOCAL_MODEL_TESTS=1 to run local model parity smoke")
	}
	cases := []struct {
		model  string
		prompt []int
		max    int
	}{
		{model: "smollm2-135m", prompt: []int{1, 2, 3, 1, 2}, max: 4},
		{model: "qwen2.5-0.5b-mlx4", prompt: []int{10, 11, 10, 11}, max: 3},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			dir := localModelPath(tc.model)
			if dir == "" {
				t.Skipf("local model %s not found", tc.model)
			}
			oldForce := ForceOnTheFly
			ForceOnTheFly = true
			defer func() { ForceOnTheFly = oldForce }()
			m, err := LoadLlama(dir)
			if err != nil {
				t.Fatalf("LoadLlama: %v", err)
			}
			normal := m.Generate(tc.prompt, tc.max)
			spec, stats := m.GenerateSpeculativeWithStats(tc.prompt, tc.max, SpeculativeConfig{Enabled: true, BlockSize: 4, NGram: 2})
			if !sameInts(spec, normal) {
				t.Fatalf("speculative output mismatch\nnormal=%v\nspec=%v", normal, spec)
			}
			if stats.VerifierBackend != "replay" {
				t.Fatalf("VerifierBackend=%q want replay", stats.VerifierBackend)
			}
			if stats.Proposer != "prompt" {
				t.Fatalf("Proposer=%q want prompt", stats.Proposer)
			}
			if stats.Steps <= 0 {
				t.Fatalf("Steps=%d want >0", stats.Steps)
			}
			if stats.ProposedTokens > 0 && stats.AcceptanceRate() < 0 {
				t.Fatalf("AcceptanceRate=%f", stats.AcceptanceRate())
			}
		})
	}
}
