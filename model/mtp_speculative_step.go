package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

// MTPSpeculativeStepResult is one internal speculative iteration: one drafter
// step, one verifier pass, and updated stats. The caller owns committing or
// restoring staged verifier KV via Verifier.Commit*KV.
type MTPSpeculativeStepResult struct {
	Draft    MTPDrafterStepResult
	Plan     MTPVerifierPlan
	Verifier MTPVerifierResult
	Stats    MTPSpeculationStats
}

// MTPMultiDraftSpeculativeResult is one internal speculative iteration with a
// bounded multi-step drafter pass followed by one verifier pass.
type MTPMultiDraftSpeculativeResult struct {
	Drafts   MTPDrafterRunResult
	Plan     MTPVerifierPlan
	Verifier MTPVerifierResult
	Stats    MTPSpeculationStats
}

// RunMTPSpeculativeStep runs one internal one-token speculative iteration. This
// is intentionally not wired to any public CLI: it is a testable integration
// seam for drafter/verifier/state/stats behavior before exposing speculation.
func (m *LlamaModel) RunMTPSpeculativeStep(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV, startPos int, kvCacheK, kvCacheV [][]float32, stats MTPSpeculationStats) (MTPSpeculativeStepResult, error) {
	multi, err := m.RunMTPMultiDraftSpeculativeStep(d, state, externalKV, startPos, 1, kvCacheK, kvCacheV, stats)
	if err != nil {
		return MTPSpeculativeStepResult{}, err
	}
	if len(multi.Drafts.Tokens) != 1 || len(multi.Drafts.Logits) != 1 || len(multi.Drafts.Activations) != 1 {
		return MTPSpeculativeStepResult{}, fmt.Errorf("MTP one-step result rows tokens/logits/activations=%d/%d/%d", len(multi.Drafts.Tokens), len(multi.Drafts.Logits), len(multi.Drafts.Activations))
	}
	draft := MTPDrafterStepResult{
		Token:          multi.Drafts.Tokens[0],
		Logits:         append([]float32(nil), multi.Drafts.Logits[0]...),
		NextActivation: append([]float32(nil), multi.Drafts.Activations[0]...),
		NextState:      multi.Drafts.FinalState,
	}
	return MTPSpeculativeStepResult{Draft: draft, Plan: multi.Plan, Verifier: multi.Verifier, Stats: multi.Stats}, nil
}

// RunMTPMultiDraftSpeculativeStep runs one internal speculative iteration with
// draftCount proposed tokens. It is not a public generation API; callers still
// own committing or restoring verifier KV after inspecting acceptance.
func (m *LlamaModel) RunMTPMultiDraftSpeculativeStep(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV, startPos, draftCount int, kvCacheK, kvCacheV [][]float32, stats MTPSpeculationStats) (MTPMultiDraftSpeculativeResult, error) {
	if m == nil {
		return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("nil model")
	}
	if draftCount <= 0 {
		return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("draft count %d out of range", draftCount)
	}
	if err := stats.ValidateOneStepCapacity(); err != nil {
		return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("MTP stats: %w", err)
	}
	drafts, err := m.RunMTPDrafterSteps(d, state, externalKV, draftCount)
	if err != nil {
		return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("MTP drafter steps: %w", err)
	}
	if len(drafts.Tokens) != draftCount {
		return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("MTP drafter produced %d tokens, want %d", len(drafts.Tokens), draftCount)
	}
	plan, err := NewMTPVerifierPlan(m, state.PreviousToken, drafts.Tokens, startPos)
	if err != nil {
		return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("MTP verifier plan: %w", err)
	}
	cp := kv.CheckpointFloatKV(kvCacheK, kvCacheV)
	verifier, err := m.RunMTPVerifierForward(plan, kvCacheK, kvCacheV)
	if err != nil {
		return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("MTP verifier forward: %w", err)
	}
	if err := stats.Record(verifier.Acceptance); err != nil {
		if restoreErr := cp.Restore(kvCacheK, kvCacheV); restoreErr != nil {
			return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("MTP stats: %w; restore staged verifier KV: %v", err, restoreErr)
		}
		return MTPMultiDraftSpeculativeResult{}, fmt.Errorf("MTP stats: %w", err)
	}
	return MTPMultiDraftSpeculativeResult{Drafts: drafts, Plan: plan, Verifier: verifier, Stats: stats}, nil
}
