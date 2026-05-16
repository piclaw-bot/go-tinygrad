package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	dir := flag.String("model", "", "model directory")
	prompt := flag.String("prompt", "The quick brown fox", "input prompt")
	tokens := flag.Int("tokens", 32, "tokens to generate")
	block := flag.Int("speculative-block", 8, "speculative proposal block size")
	ngram := flag.Int("speculative-ngram", 4, "speculative prompt-lookup n-gram size")
	minProposal := flag.Int("speculative-min-proposal", 2, "minimum proposal length before verifier attempt")
	proposer := flag.String("speculative-proposer", "prompt", "speculative proposer: prompt, none")
	outPath := flag.String("csv", "", "optional CSV output path")
	flag.Parse()

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: specbench -model <dir> [-prompt text] [-tokens N]")
		os.Exit(1)
	}
	if *tokens < 0 {
		fmt.Fprintln(os.Stderr, "tokens must be non-negative")
		os.Exit(1)
	}

	m, err := model.LoadLlama(*dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	tok, err := tokenizer.Load(*dir + "/tokenizer.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(1)
	}
	m.Tok = tok
	ids := tok.Encode(*prompt)

	normalStart := time.Now()
	normal := m.Generate(ids, *tokens)
	normalElapsed := time.Since(normalStart)

	cfg := model.SpeculativeConfig{
		Enabled:     true,
		BlockSize:   *block,
		NGram:       *ngram,
		MinProposal: *minProposal,
		Proposer:    *proposer,
	}.Normalize()
	specStart := time.Now()
	spec, stats := m.GenerateSpeculativeWithStats(ids, *tokens, cfg)
	specElapsed := time.Since(specStart)

	match := sameInts(normal, spec)
	rows := [][]string{{
		"model", "prompt_tokens", "max_tokens", "mode", "elapsed_ms", "tokens_per_sec", "match_normal",
		"backend", "proposer", "steps", "proposal_steps", "proposed", "accepted", "bonus", "fallback", "acceptance", "emitted", "tokens_per_step", "avg_proposal",
	}}
	modelID := baseName(*dir)
	rows = append(rows, benchRow(modelID, len(ids), *tokens, "normal", normalElapsed, len(normal)-len(ids), true, model.SpeculativeStats{}))
	rows = append(rows, benchRow(modelID, len(ids), *tokens, "speculative", specElapsed, len(spec)-len(ids), match, stats))

	w := csv.NewWriter(os.Stdout)
	if *outPath != "" {
		f, err := os.Create(*outPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "csv create: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		w = csv.NewWriter(f)
	}
	if err := w.WriteAll(rows); err != nil {
		fmt.Fprintf(os.Stderr, "csv write: %v\n", err)
		os.Exit(1)
	}
}

func benchRow(modelID string, promptTokens, maxTokens int, mode string, elapsed time.Duration, generated int, match bool, stats model.SpeculativeStats) []string {
	tokS := 0.0
	if elapsed > 0 {
		tokS = float64(generated) / elapsed.Seconds()
	}
	return []string{
		modelID,
		strconv.Itoa(promptTokens),
		strconv.Itoa(maxTokens),
		mode,
		strconv.FormatInt(elapsed.Milliseconds(), 10),
		strconv.FormatFloat(tokS, 'f', 3, 64),
		strconv.FormatBool(match),
		stats.VerifierBackend,
		stats.Proposer,
		strconv.Itoa(stats.Steps),
		strconv.Itoa(stats.ProposalSteps),
		strconv.Itoa(stats.ProposedTokens),
		strconv.Itoa(stats.AcceptedTokens),
		strconv.Itoa(stats.BonusTokens),
		strconv.Itoa(stats.FallbackSteps),
		strconv.FormatFloat(stats.AcceptanceRate(), 'f', 3, 64),
		strconv.Itoa(stats.EmittedTokens()),
		strconv.FormatFloat(stats.TokensPerStep(), 'f', 3, 64),
		strconv.FormatFloat(stats.AverageProposalLen(), 'f', 3, 64),
	}
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
