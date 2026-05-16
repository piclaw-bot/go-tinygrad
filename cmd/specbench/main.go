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
	backend := flag.String("speculative-backend", "replay", "speculative verifier backend: replay")
	repeat := flag.Int("repeat", 1, "number of timed runs to average")
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
	if *repeat <= 0 {
		fmt.Fprintln(os.Stderr, "repeat must be positive")
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

	cfg := model.SpeculativeConfig{
		Enabled:     true,
		BlockSize:   *block,
		NGram:       *ngram,
		MinProposal: *minProposal,
		Proposer:    *proposer,
		Backend:     *backend,
	}.Normalize()

	var normal, spec []int
	var normalElapsed, specElapsed time.Duration
	var stats model.SpeculativeStats
	match := true
	for i := 0; i < *repeat; i++ {
		normalStart := time.Now()
		runNormal := m.Generate(ids, *tokens)
		normalElapsed += time.Since(normalStart)

		specStart := time.Now()
		runSpec, runStats := m.GenerateSpeculativeWithStats(ids, *tokens, cfg)
		specElapsed += time.Since(specStart)

		if i == 0 {
			normal = runNormal
			spec = runSpec
		} else {
			match = match && sameInts(normal, runNormal) && sameInts(spec, runSpec)
		}
		match = match && sameInts(runNormal, runSpec)
		stats = stats.Add(runStats)
	}
	normalElapsed /= time.Duration(*repeat)
	specElapsed /= time.Duration(*repeat)
	stats = stats.Average(*repeat)

	normalTokS := tokensPerSecond(len(normal)-len(ids), normalElapsed)
	specTokS := tokensPerSecond(len(spec)-len(ids), specElapsed)
	speedup := 0.0
	if normalTokS > 0 {
		speedup = specTokS / normalTokS
	}
	rows := [][]string{{
		"model", "prompt_tokens", "max_tokens", "repeat", "mode", "elapsed_ms", "tokens_per_sec", "speedup_vs_normal", "match_normal",
		"backend", "proposer", "steps", "proposal_steps", "proposed", "accepted", "bonus", "fallback", "acceptance", "emitted", "tokens_per_step", "avg_proposal",
	}}
	modelID := baseName(*dir)
	rows = append(rows, benchRow(modelID, len(ids), *tokens, *repeat, "normal", normalElapsed, len(normal)-len(ids), 1.0, true, model.SpeculativeStats{}))
	rows = append(rows, benchRow(modelID, len(ids), *tokens, *repeat, "speculative", specElapsed, len(spec)-len(ids), speedup, match, stats))

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

func benchRow(modelID string, promptTokens, maxTokens, repeat int, mode string, elapsed time.Duration, generated int, speedup float64, match bool, stats model.SpeculativeStats) []string {
	tokS := tokensPerSecond(generated, elapsed)
	return []string{
		modelID,
		strconv.Itoa(promptTokens),
		strconv.Itoa(maxTokens),
		strconv.Itoa(repeat),
		mode,
		strconv.FormatInt(elapsed.Milliseconds(), 10),
		strconv.FormatFloat(tokS, 'f', 3, 64),
		strconv.FormatFloat(speedup, 'f', 3, 64),
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

func tokensPerSecond(generated int, elapsed time.Duration) float64 {
	if elapsed <= 0 {
		return 0
	}
	return float64(generated) / elapsed.Seconds()
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
