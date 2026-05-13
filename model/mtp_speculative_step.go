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

// RunMTPSpeculativeStep runs one internal one-token speculative iteration. This
// is intentionally not wired to any public CLI: it is a testable integration
// seam for drafter/verifier/state/stats behavior before exposing speculation.
func (m *LlamaModel) RunMTPSpeculativeStep(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV, startPos int, kvCacheK, kvCacheV [][]float32, stats MTPSpeculationStats) (MTPSpeculativeStepResult, error) {
	if m == nil {
		return MTPSpeculativeStepResult{}, fmt.Errorf("nil model")
	}
	if err := stats.ValidateOneStepCapacity(); err != nil {
		return MTPSpeculativeStepResult{}, fmt.Errorf("MTP stats: %w", err)
	}
	draft, err := m.RunMTPDrafterStepWithExternalKV(d, state, externalKV)
	if err != nil {
		return MTPSpeculativeStepResult{}, fmt.Errorf("MTP drafter step: %w", err)
	}
	plan, err := NewMTPVerifierPlan(m, state.PreviousToken, []int{draft.Token}, startPos)
	if err != nil {
		return MTPSpeculativeStepResult{}, fmt.Errorf("MTP verifier plan: %w", err)
	}
	cp := kv.CheckpointFloatKV(kvCacheK, kvCacheV)
	verifier, err := m.RunMTPVerifierForward(plan, kvCacheK, kvCacheV)
	if err != nil {
		return MTPSpeculativeStepResult{}, fmt.Errorf("MTP verifier forward: %w", err)
	}
	if err := stats.Record(verifier.Acceptance); err != nil {
		if restoreErr := cp.Restore(kvCacheK, kvCacheV); restoreErr != nil {
			return MTPSpeculativeStepResult{}, fmt.Errorf("MTP stats: %w; restore staged verifier KV: %v", err, restoreErr)
		}
		return MTPSpeculativeStepResult{}, fmt.Errorf("MTP stats: %w", err)
	}
	return MTPSpeculativeStepResult{Draft: draft, Plan: plan, Verifier: verifier, Stats: stats}, nil
}
