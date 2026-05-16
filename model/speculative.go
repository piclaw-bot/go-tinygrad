package model

import "os"

// SpeculativeConfig controls the opt-in stock-weight speculative decoding path.
// It intentionally does not depend on Orthrus custom diffusion weights: the
// proposer is pluggable and the verifier remains the normal model.
type SpeculativeConfig struct {
	Enabled   bool
	BlockSize int
	NGram     int
}

func SpeculativeConfigFromEnv() SpeculativeConfig {
	cfg := SpeculativeConfig{Enabled: os.Getenv("GO_PHERENCE_SPECULATIVE") == "1", BlockSize: 8, NGram: 4}
	if cfg.BlockSize <= 0 {
		cfg.BlockSize = 8
	}
	if cfg.NGram <= 0 {
		cfg.NGram = 4
	}
	return cfg
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
	return m.Generate(tokenIDs, maxTokens)
}
