package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
)

type CheckResult struct {
	PromptIndex      int                    `json:"prompt_index"`
	PromptTokens     int                    `json:"prompt_tokens"`
	MaxTokens        int                    `json:"max_tokens"`
	Proposer         string                 `json:"proposer"`
	Backend          string                 `json:"backend"`
	Match            bool                   `json:"match"`
	NormalLen        int                    `json:"normal_len"`
	SpeculativeLen   int                    `json:"speculative_len"`
	FirstMismatch    int                    `json:"first_mismatch"`
	NormalToken      int                    `json:"normal_token,omitempty"`
	SpeculativeToken int                    `json:"speculative_token,omitempty"`
	Stats            model.SpeculativeStats `json:"stats"`
}

type CheckReport struct {
	Model   string        `json:"model"`
	Passed  bool          `json:"passed"`
	Results []CheckResult `json:"results"`
}

func main() {
	dir := flag.String("model", "", "model directory")
	prompt := flag.String("prompt", "The quick brown fox", "input prompt")
	promptFile := flag.String("prompt-file", "", "optional newline-delimited prompt file")
	tokens := flag.Int("tokens", 32, "tokens to generate")
	block := flag.Int("speculative-block", 8, "speculative proposal block size")
	ngram := flag.Int("speculative-ngram", 4, "speculative prompt-lookup n-gram size")
	minProposal := flag.Int("speculative-min-proposal", 2, "minimum proposal length before verifier attempt")
	proposerList := flag.String("proposers", "prompt,repeat-last,none", "comma-separated proposer list")
	backend := flag.String("speculative-backend", "replay", "speculative verifier backend")
	flag.Parse()

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: speccheck -model <dir> [-prompt text|-prompt-file file] [-tokens N]")
		os.Exit(2)
	}
	if *tokens < 0 {
		fmt.Fprintln(os.Stderr, "tokens must be non-negative")
		os.Exit(2)
	}
	m, err := model.LoadLlama(*dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(2)
	}
	tok, err := tokenizer.Load(*dir + "/tokenizer.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(2)
	}
	m.Tok = tok
	prompts, err := loadPrompts(*prompt, *promptFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "prompts: %v\n", err)
		os.Exit(2)
	}
	proposers := parseList(*proposerList)
	if len(proposers) == 0 {
		fmt.Fprintln(os.Stderr, "no proposers selected")
		os.Exit(2)
	}

	report := CheckReport{Model: baseName(*dir), Passed: true}
	for pi, promptText := range prompts {
		ids := tok.Encode(promptText)
		normal := m.Generate(ids, *tokens)
		preparedLen := len(m.PreparedGenerateTokens(ids))
		for _, proposer := range proposers {
			cfg := model.SpeculativeConfig{
				Enabled:     true,
				BlockSize:   *block,
				NGram:       *ngram,
				MinProposal: *minProposal,
				Proposer:    proposer,
				Backend:     *backend,
			}.Normalize()
			spec, stats := m.GenerateSpeculativeWithStats(ids, *tokens, cfg)
			idx := firstMismatch(normal, spec)
			res := CheckResult{
				PromptIndex:    pi,
				PromptTokens:   preparedLen,
				MaxTokens:      *tokens,
				Proposer:       cfg.Proposer,
				Backend:        stats.VerifierBackend,
				Match:          idx < 0,
				NormalLen:      len(normal),
				SpeculativeLen: len(spec),
				FirstMismatch:  idx,
				Stats:          stats,
			}
			if idx >= 0 {
				report.Passed = false
				if idx < len(normal) {
					res.NormalToken = normal[idx]
				}
				if idx < len(spec) {
					res.SpeculativeToken = spec[idx]
				}
			}
			report.Results = append(report.Results, res)
		}
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintf(os.Stderr, "json: %v\n", err)
		os.Exit(2)
	}
	if !report.Passed {
		os.Exit(1)
	}
}

func firstMismatch(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

func parseList(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func loadPrompts(prompt, promptFile string) ([]string, error) {
	if promptFile == "" {
		return []string{prompt}, nil
	}
	f, err := os.Open(promptFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var prompts []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		prompts = append(prompts, line)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(prompts) == 0 {
		return nil, fmt.Errorf("no prompts in %s", promptFile)
	}
	return prompts, nil
}

func baseName(path string) string {
	for len(path) > 0 && (path[len(path)-1] == '/' || path[len(path)-1] == '\\') {
		path = path[:len(path)-1]
	}
	last := 0
	for i := range path {
		if path[i] == '/' || path[i] == '\\' {
			last = i + 1
		}
	}
	return path[last:]
}
