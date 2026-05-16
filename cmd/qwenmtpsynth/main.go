package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/model"
)

type Report struct {
	Passed     bool                     `json:"passed"`
	Drafted    []int                    `json:"drafted"`
	Acceptance model.MTPAcceptance      `json:"acceptance"`
	Stats      model.QwenNativeMTPStats `json:"stats"`
}

func main() {
	steps := flag.Int("steps", 2, "synthetic MTP draft steps")
	flag.Parse()
	if *steps < 0 {
		fmt.Fprintln(os.Stderr, "steps must be non-negative")
		os.Exit(2)
	}
	m, head, meta, state := model.NewSyntheticQwenNativeMTPFixture()
	plan, err := model.NewQwenNativeMTPPlan(0, state, *steps, meta)
	if err != nil {
		fmt.Fprintf(os.Stderr, "plan: %v\n", err)
		os.Exit(2)
	}
	_, drafted, _, err := head.DraftSteps(m, plan.TokenID, plan.State, plan.MaxSteps, 1e-6, meta)
	if err != nil {
		fmt.Fprintf(os.Stderr, "draft: %v\n", err)
		os.Exit(2)
	}
	verifier := append([]int(nil), drafted...)
	verifier = append(verifier, 1)
	res, err := model.RunQwenNativeMTPPlan(head, m, plan, verifier, 1e-6, meta)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run: %v\n", err)
		os.Exit(2)
	}
	report := Report{Passed: res.Acceptance.AllDraftsAccepted, Drafted: res.Drafted, Acceptance: res.Acceptance, Stats: res.Stats}
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
