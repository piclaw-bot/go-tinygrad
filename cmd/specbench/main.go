package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	dir := flag.String("model", "", "model directory")
	prompt := flag.String("prompt", "The quick brown fox", "input prompt")
	promptFile := flag.String("prompt-file", "", "optional newline-delimited prompt file")
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
	prompts, err := loadPrompts(*prompt, *promptFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "prompts: %v\n", err)
		os.Exit(1)
	}

	cfg := model.SpeculativeConfig{
		Enabled:     true,
		BlockSize:   *block,
		NGram:       *ngram,
		MinProposal: *minProposal,
		Proposer:    *proposer,
		Backend:     *backend,
	}.Normalize()

	type result struct {
		promptIdx       int
		promptTokens    int
		normalGenerated int
		specGenerated   int
		normalElapsed   time.Duration
		specElapsed     time.Duration
		match           bool
		stats           model.SpeculativeStats
	}
	results := make([]result, 0, len(prompts))
	for pi, promptText := range prompts {
		ids := tok.Encode(promptText)
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
		results = append(results, result{
			promptIdx:       pi,
			promptTokens:    len(ids),
			normalGenerated: len(normal) - len(ids),
			specGenerated:   len(spec) - len(ids),
			normalElapsed:   normalElapsed,
			specElapsed:     specElapsed,
			match:           match,
			stats:           stats,
		})
	}
	rows := [][]string{{
		"model", "prompt_index", "prompt_tokens", "max_tokens", "repeat", "mode", "elapsed_ms", "tokens_per_sec", "speedup_vs_normal", "match_normal",
		"backend", "proposer", "steps", "proposal_steps", "proposed", "accepted", "bonus", "fallback", "acceptance", "emitted", "tokens_per_step", "avg_proposal",
	}}
	modelID := baseName(*dir)
	var totalPromptTokens, totalNormalGenerated, totalSpecGenerated int
	var totalNormalElapsed, totalSpecElapsed time.Duration
	aggregateMatch := true
	var aggregateStats model.SpeculativeStats
	for _, res := range results {
		normalTokS := tokensPerSecond(res.normalGenerated, res.normalElapsed)
		specTokS := tokensPerSecond(res.specGenerated, res.specElapsed)
		speedup := 0.0
		if normalTokS > 0 {
			speedup = specTokS / normalTokS
		}
		rows = append(rows, benchRow(modelID, res.promptIdx, res.promptTokens, *tokens, *repeat, "normal", res.normalElapsed, res.normalGenerated, 1.0, true, model.SpeculativeStats{}))
		rows = append(rows, benchRow(modelID, res.promptIdx, res.promptTokens, *tokens, *repeat, "speculative", res.specElapsed, res.specGenerated, speedup, res.match, res.stats))
		totalPromptTokens += res.promptTokens
		totalNormalGenerated += res.normalGenerated
		totalSpecGenerated += res.specGenerated
		totalNormalElapsed += res.normalElapsed
		totalSpecElapsed += res.specElapsed
		aggregateMatch = aggregateMatch && res.match
		aggregateStats = aggregateStats.Add(res.stats)
	}
	if len(results) > 1 {
		normalTokS := tokensPerSecond(totalNormalGenerated, totalNormalElapsed)
		specTokS := tokensPerSecond(totalSpecGenerated, totalSpecElapsed)
		speedup := 0.0
		if normalTokS > 0 {
			speedup = specTokS / normalTokS
		}
		rows = append(rows, benchRow(modelID, -1, totalPromptTokens, *tokens, *repeat, "normal_total", totalNormalElapsed, totalNormalGenerated, 1.0, true, model.SpeculativeStats{}))
		rows = append(rows, benchRow(modelID, -1, totalPromptTokens, *tokens, *repeat, "speculative_total", totalSpecElapsed, totalSpecGenerated, speedup, aggregateMatch, aggregateStats))
	}

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

func benchRow(modelID string, promptIdx, promptTokens, maxTokens, repeat int, mode string, elapsed time.Duration, generated int, speedup float64, match bool, stats model.SpeculativeStats) []string {
	tokS := tokensPerSecond(generated, elapsed)
	return []string{
		modelID,
		strconv.Itoa(promptIdx),
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
