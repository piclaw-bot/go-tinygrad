package model

import (
	"fmt"
	"os"
	"strconv"
)

// SpeculativeConfig controls the opt-in stock-weight speculative decoding path.
// It intentionally does not depend on Orthrus custom diffusion weights: the
// proposer is pluggable and the verifier remains the normal model.
type SpeculativeConfig struct {
	Enabled   bool
	BlockSize int
	NGram     int
	Proposer  string
	Debug     bool
}

type SpeculativeProposer interface {
	Name() string
	Propose(context []int, max int) []int
}

type NoopProposer struct{}

func (NoopProposer) Name() string { return "none" }

func (NoopProposer) Propose(context []int, max int) []int { return nil }

type PromptLookupProposer struct {
	NGram int
}

func (p PromptLookupProposer) Name() string { return "prompt" }

func (p PromptLookupProposer) Propose(context []int, max int) []int {
	return PromptLookupProposal(context, max, p.NGram)
}

func NewSpeculativeProposer(cfg SpeculativeConfig) SpeculativeProposer {
	switch cfg.Proposer {
	case "none", "off", "disabled":
		return NoopProposer{}
	case "", "prompt", "prompt-lookup", "ngram":
		return PromptLookupProposer{NGram: cfg.NGram}
	default:
		return PromptLookupProposer{NGram: cfg.NGram}
	}
}

type SpeculativeStats struct {
	VerifierBackend string
	Proposer        string
	Steps           int
	ProposalSteps   int
	ProposedTokens  int
	AcceptedTokens  int
	BonusTokens     int
	FallbackSteps   int
}

func (s SpeculativeStats) AcceptanceRate() float64 {
	if s.ProposedTokens <= 0 {
		return 0
	}
	return float64(s.AcceptedTokens) / float64(s.ProposedTokens)
}

func SpeculativeConfigFromEnv() SpeculativeConfig {
	cfg := SpeculativeConfig{
		Enabled:   os.Getenv("GO_PHERENCE_SPECULATIVE") == "1",
		BlockSize: envPositiveInt("GO_PHERENCE_SPECULATIVE_BLOCK", 8),
		NGram:     envPositiveInt("GO_PHERENCE_SPECULATIVE_NGRAM", 4),
		Proposer:  envString("GO_PHERENCE_SPECULATIVE_PROPOSER", "prompt"),
		Debug:     os.Getenv("GO_PHERENCE_SPECULATIVE_DEBUG") == "1",
	}
	return cfg
}

func envString(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}

func envPositiveInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// PromptLookupProposal proposes up to max tokens by finding the longest suffix
// of context that appears earlier in the same context and copying the following
// tokens. This is exact only after the normal verifier accepts it; by itself it
// is just a cheap stock-weight proposer.
func PromptLookupProposal(context []int, max int, maxNGram int) []int {
	if max <= 0 || len(context) < 2 {
		return nil
	}
	if maxNGram <= 0 || maxNGram > len(context)-1 {
		maxNGram = len(context) - 1
	}
	for n := maxNGram; n >= 1; n-- {
		start := len(context) - n
		best := -1
		for i := 0; i+n < len(context); i++ {
			match := true
			for j := 0; j < n; j++ {
				if context[i+j] != context[start+j] {
					match = false
					break
				}
			}
			if match {
				best = i
			}
		}
		if best >= 0 {
			avail := len(context) - (best + n)
			if avail <= 0 {
				continue
			}
			if avail > max {
				avail = max
			}
			out := make([]int, avail)
			copy(out, context[best+n:best+n+avail])
			return out
		}
	}
	return nil
}

// GenerateSpeculative is the opt-in entry point for stock-weight speculative
// decoding. The verifier/block execution will be filled in behind this API; for
// now it preserves exact behavior by falling back to normal generation when no
// verified block path is available.
func (m *LlamaModel) GenerateSpeculative(tokenIDs []int, maxTokens int, cfg SpeculativeConfig) []int {
	if !cfg.Enabled || maxTokens <= 0 {
		return m.Generate(tokenIDs, maxTokens)
	}
	prepared := m.prepareGenerateTokens(tokenIDs)
	if maxTokens < 0 {
		return append([]int(nil), prepared...)
	}
	maxInt := int(^uint(0) >> 1)
	if maxTokens > maxInt-len(prepared) {
		return append([]int(nil), prepared...)
	}
	proposer := NewSpeculativeProposer(cfg)
	state, err := NewCPUDecodeStateForSpeculative(m, prepared, maxTokens)
	if err != nil {
		return m.generatePrepared(prepared, maxTokens)
	}
	stats := SpeculativeStats{VerifierBackend: state.VerifierBackend(), Proposer: proposer.Name()}
	defer func() {
		if cfg.Debug {
			fmt.Fprintf(os.Stderr, "speculative backend=%s proposer=%s steps=%d proposal_steps=%d proposed=%d accepted=%d bonus=%d fallback=%d acceptance=%.2f\n",
				stats.VerifierBackend, stats.Proposer, stats.Steps, stats.ProposalSteps, stats.ProposedTokens, stats.AcceptedTokens, stats.BonusTokens, stats.FallbackSteps, stats.AcceptanceRate())
		}
	}()
	for len(state.Output) < len(prepared)+maxTokens {
		stats.Steps++
		remaining := len(prepared) + maxTokens - len(state.Output)
		block := cfg.BlockSize
		if block <= 0 || block > remaining-1 {
			block = remaining - 1
		}
		proposal := proposer.Propose(state.Output, block)
		if len(proposal) == 0 {
			stats.FallbackSteps++
			if _, err := state.DecodeOneGreedy(); err != nil {
				return state.Output
			}
			continue
		}

		// Greedy verifier: run the real model for the proposed block plus one
		// bonus token, then accept the longest matching prefix. This is exact; the
		// verifier backend is hidden behind CPUDecodeState so it can be replaced
		// with a KV-reusing implementation without changing this loop.
		stats.ProposalSteps++
		stats.ProposedTokens += len(proposal)
		checkpoint := state.Checkpoint()
		acceptance, err := state.VerifyGreedyBlock(proposal)
		if err != nil {
			stats.FallbackSteps++
			_ = state.Restore(checkpoint)
			if _, err := state.DecodeOneGreedy(); err != nil {
				return state.Output
			}
			continue
		}
		stats.AcceptedTokens += acceptance.AcceptedPrefixLen
		stats.BonusTokens++
		if err := state.CommitAcceptedOutputOnly(checkpoint, acceptance); err != nil {
			_ = state.Restore(checkpoint)
			if _, err := state.DecodeOneGreedy(); err != nil {
				return state.Output
			}
		}
		if len(state.Output) > len(prepared)+maxTokens {
			state.Output = state.Output[:len(prepared)+maxTokens]
		}
	}
	return state.Output
}
