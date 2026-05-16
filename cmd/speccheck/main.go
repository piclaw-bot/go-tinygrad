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
	Model              string        `json:"model"`
	Passed             bool          `json:"passed"`
	GoldenMatch        bool          `json:"golden_match,omitempty"`
	TotalChecks        int           `json:"total_checks"`
	FailedChecks       int           `json:"failed_checks"`
	GoldenChecks       int           `json:"golden_checks,omitempty"`
	FailedGoldenChecks int           `json:"failed_golden_checks,omitempty"`
	Results            []CheckResult `json:"results"`
}

type GoldenReport struct {
	Model   string         `json:"model"`
	Prompts []GoldenPrompt `json:"prompts"`
}

type GoldenPrompt struct {
	PromptIndex  int   `json:"prompt_index"`
	PromptTokens int   `json:"prompt_tokens"`
	MaxTokens    int   `json:"max_tokens"`
	Output       []int `json:"output"`
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
	nativeQwenMTP := flag.Bool("qwen-native-mtp", false, "check Qwen3.5/Qwen3.6 native MTP path when available")
	goldenPath := flag.String("golden", "", "optional golden JSON to compare normal outputs against")
	writeGoldenPath := flag.String("write-golden", "", "optional path to write normal-output golden JSON")
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
	if *nativeQwenMTP {
		fmt.Fprintln(os.Stderr, "qwen native MTP speccheck mode is available only for synthetic unit fixtures until Qwen3.6 LoadLlama support lands")
		os.Exit(2)
	}
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

	modelID := baseName(*dir)
	golden, err := loadGolden(*goldenPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "golden: %v\n", err)
		os.Exit(2)
	}
	report := CheckReport{Model: modelID, Passed: true, GoldenMatch: true}
	if golden != nil {
		report.GoldenChecks += 2 // model metadata + prompt-count metadata
		if golden.Model != "" && golden.Model != modelID {
			report.Passed = false
			report.GoldenMatch = false
			report.FailedGoldenChecks++
			fmt.Fprintf(os.Stderr, "golden model mismatch: got %q want %q\n", modelID, golden.Model)
		}
		if len(golden.Prompts) != len(prompts) {
			report.Passed = false
			report.GoldenMatch = false
			report.FailedGoldenChecks++
			fmt.Fprintf(os.Stderr, "golden prompt count mismatch: got %d want %d\n", len(prompts), len(golden.Prompts))
		}
	}
	writeGolden := GoldenReport{Model: modelID}
	for pi, promptText := range prompts {
		ids := tok.Encode(promptText)
		normal := m.Generate(ids, *tokens)
		preparedLen := len(m.PreparedGenerateTokens(ids))
		writeGolden.Prompts = append(writeGolden.Prompts, GoldenPrompt{PromptIndex: pi, PromptTokens: preparedLen, MaxTokens: *tokens, Output: append([]int(nil), normal...)})
		if golden != nil {
			report.GoldenChecks++
			if err := compareGoldenPrompt(golden, pi, preparedLen, *tokens, normal); err != nil {
				report.Passed = false
				report.GoldenMatch = false
				report.FailedGoldenChecks++
				fmt.Fprintf(os.Stderr, "golden mismatch prompt %d: %v\n", pi, err)
			}
		}
		for _, proposer := range proposers {
			report.TotalChecks++
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
				report.FailedChecks++
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

	if *writeGoldenPath != "" {
		if err := writeGoldenFile(*writeGoldenPath, writeGolden); err != nil {
			fmt.Fprintf(os.Stderr, "write golden: %v\n", err)
			os.Exit(2)
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

func loadGolden(path string) (*GoldenReport, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var g GoldenReport
	if err := json.NewDecoder(f).Decode(&g); err != nil {
		return nil, err
	}
	return &g, nil
}

func writeGoldenFile(path string, golden GoldenReport) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(golden)
}

func compareGoldenPrompt(g *GoldenReport, promptIdx, promptTokens, maxTokens int, normal []int) error {
	if g == nil {
		return nil
	}
	if promptIdx < 0 || promptIdx >= len(g.Prompts) {
		return fmt.Errorf("golden has %d prompts", len(g.Prompts))
	}
	gp := g.Prompts[promptIdx]
	if gp.PromptIndex != promptIdx || gp.PromptTokens != promptTokens || gp.MaxTokens != maxTokens {
		return fmt.Errorf("metadata got idx/tokens/max=%d/%d/%d want %d/%d/%d", promptIdx, promptTokens, maxTokens, gp.PromptIndex, gp.PromptTokens, gp.MaxTokens)
	}
	if idx := firstMismatch(gp.Output, normal); idx >= 0 {
		return fmt.Errorf("output mismatch at %d", idx)
	}
	return nil
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
