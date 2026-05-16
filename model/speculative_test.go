package model

import "testing"

func TestPromptLookupProposer(t *testing.T) {
	p := PromptLookupProposer{NGram: 2}
	got := p.Propose([]int{1, 2, 3, 4, 1, 2}, 2)
	want := []int{3, 4}
	if !sameInts(got, want) {
		t.Fatalf("proposal=%v want %v", got, want)
	}
}

func TestPromptLookupProposal(t *testing.T) {
	ctx := []int{1, 2, 3, 4, 1, 2}
	got := PromptLookupProposal(ctx, 2, 2)
	want := []int{3, 4}
	if !sameInts(got, want) {
		t.Fatalf("proposal=%v want %v", got, want)
	}
}

func TestPromptLookupProposalNoMatch(t *testing.T) {
	if got := PromptLookupProposal([]int{1, 2, 3}, 4, 2); got != nil {
		t.Fatalf("proposal=%v want nil", got)
	}
}

func TestPromptLookupProposalBounds(t *testing.T) {
	ctx := []int{7, 8, 9, 7, 8}
	got := PromptLookupProposal(ctx, 1, 99)
	want := []int{9}
	if !sameInts(got, want) {
		t.Fatalf("proposal=%v want %v", got, want)
	}
	if got := PromptLookupProposal(ctx, 0, 2); got != nil {
		t.Fatalf("zero max proposal=%v want nil", got)
	}
}
