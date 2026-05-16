package model

import "testing"

func TestGenerateSpeculativeWithStatsDisabled(t *testing.T) {
	m := &LlamaModel{}
	out, stats := m.GenerateSpeculativeWithStats([]int{1, 2}, -1, SpeculativeConfig{})
	if !sameInts(out, []int{1, 2}) {
		t.Fatalf("out=%v want prompt", out)
	}
	if stats.Steps != 0 || stats.ProposalSteps != 0 {
		t.Fatalf("stats=%+v want zero", stats)
	}
}

func TestSpeculativeStatsDerivedMetrics(t *testing.T) {
	stats := SpeculativeStats{VerifierBackend: "replay", Steps: 3, ProposalSteps: 2, ProposedTokens: 4, AcceptedTokens: 3, BonusTokens: 2, FallbackSteps: 1}
	if got := stats.AcceptanceRate(); got != 0.75 {
		t.Fatalf("AcceptanceRate=%v want 0.75", got)
	}
	if got := stats.EmittedTokens(); got != 6 {
		t.Fatalf("EmittedTokens=%d want 6", got)
	}
	if got := stats.TokensPerStep(); got != 2.0 {
		t.Fatalf("TokensPerStep=%v want 2", got)
	}
	if got := stats.AverageProposalLen(); got != 2.0 {
		t.Fatalf("AverageProposalLen=%v want 2", got)
	}
	stats = SpeculativeStats{}
	if stats.AcceptanceRate() != 0 || stats.TokensPerStep() != 0 || stats.AverageProposalLen() != 0 {
		t.Fatalf("zero stats produced non-zero derived metrics: %+v", stats)
	}
}

func TestSpeculativeConfigNormalize(t *testing.T) {
	cfg := (SpeculativeConfig{}).Normalize()
	if cfg.BlockSize != 8 || cfg.NGram != 4 || cfg.MinProposal != 2 || cfg.Proposer != "prompt" || cfg.Backend != "replay" {
		t.Fatalf("Normalize=%+v, want safe defaults", cfg)
	}
	cfg = (SpeculativeConfig{BlockSize: 1, NGram: 2, MinProposal: 3, Proposer: "none", Backend: "replay"}).Normalize()
	if cfg.BlockSize != 1 || cfg.NGram != 2 || cfg.MinProposal != 3 || cfg.Proposer != "none" || cfg.Backend != "replay" {
		t.Fatalf("Normalize changed explicit values: %+v", cfg)
	}
}

func TestSpeculativeConfigMinProposalDefault(t *testing.T) {
	t.Setenv("GO_PHERENCE_SPECULATIVE", "1")
	cfg := SpeculativeConfigFromEnv()
	if cfg.MinProposal != 2 {
		t.Fatalf("MinProposal=%d want 2", cfg.MinProposal)
	}
	t.Setenv("GO_PHERENCE_SPECULATIVE_MIN_PROPOSAL", "3")
	cfg = SpeculativeConfigFromEnv()
	if cfg.MinProposal != 3 {
		t.Fatalf("MinProposal=%d want 3", cfg.MinProposal)
	}
}

func TestNoopProposer(t *testing.T) {
	p := NewSpeculativeProposer(SpeculativeConfig{Proposer: "none"})
	if p.Name() != "none" {
		t.Fatalf("Name=%q want none", p.Name())
	}
	if got := p.Propose([]int{1, 2, 1}, 4); got != nil {
		t.Fatalf("noop proposal=%v want nil", got)
	}
}

func TestNewSpeculativeProposerDefaultsToPromptLookup(t *testing.T) {
	p := NewSpeculativeProposer(SpeculativeConfig{Proposer: "unknown", NGram: 2})
	got := p.Propose([]int{1, 2, 3, 1, 2}, 1)
	if !sameInts(got, []int{3}) {
		t.Fatalf("proposal=%v want [3]", got)
	}
}

func TestPromptLookupProposer(t *testing.T) {
	p := PromptLookupProposer{NGram: 2}
	if p.Name() != "prompt" {
		t.Fatalf("Name=%q want prompt", p.Name())
	}
	got := p.Propose([]int{1, 2, 3, 4, 1, 2}, 2)
	want := []int{3, 4}
	if !sameInts(got, want) {
		t.Fatalf("proposal=%v want %v", got, want)
	}
}

func TestPromptLookupProposal(t *testing.T) {
	ctx := []int{1, 2, 3, 4, 1, 2}
	got := PromptLookupProposal(ctx, 2, 2)
	want := []int{3, 4}
	if !sameInts(got, want) {
		t.Fatalf("proposal=%v want %v", got, want)
	}
}

func TestPromptLookupProposalNoMatch(t *testing.T) {
	if got := PromptLookupProposal([]int{1, 2, 3}, 4, 2); got != nil {
		t.Fatalf("proposal=%v want nil", got)
	}
}

func TestPromptLookupProposalBounds(t *testing.T) {
	ctx := []int{7, 8, 9, 7, 8}
	got := PromptLookupProposal(ctx, 1, 99)
	want := []int{9}
	if !sameInts(got, want) {
		t.Fatalf("proposal=%v want %v", got, want)
	}
	if got := PromptLookupProposal(ctx, 0, 2); got != nil {
		t.Fatalf("zero max proposal=%v want nil", got)
	}
}
