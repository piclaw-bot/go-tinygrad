package model

import "testing"

func TestAcceptMTPDraftAllAcceptedUsesFinalVerifierBonus(t *testing.T) {
	got, err := AcceptMTPDraft([]int{10, 11, 12}, []int{10, 11, 12, 99})
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	if !got.AllDraftsAccepted {
		t.Fatal("AllDraftsAccepted=false, want true")
	}
	if got.AcceptedPrefixLen != 3 || got.VerifiedCount != 3 || got.DraftedCount != 3 {
		t.Fatalf("counts = accepted %d verified %d drafted %d, want 3/3/3", got.AcceptedPrefixLen, got.VerifiedCount, got.DraftedCount)
	}
	if got.FirstRejectedIndex != -1 {
		t.Fatalf("FirstRejectedIndex=%d, want -1", got.FirstRejectedIndex)
	}
	if got.BonusToken != 99 {
		t.Fatalf("BonusToken=%d, want 99", got.BonusToken)
	}
	if !sameInts(got.AcceptedTokens, []int{10, 11, 12}) || !sameInts(got.OutputTokens, []int{10, 11, 12, 99}) {
		t.Fatalf("accepted=%v output=%v", got.AcceptedTokens, got.OutputTokens)
	}
}

func TestAcceptMTPDraftMismatchUsesVerifierMismatchAsBonus(t *testing.T) {
	got, err := AcceptMTPDraft([]int{10, 11, 12, 13}, []int{10, 11, 42, 43, 44})
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	if got.AllDraftsAccepted {
		t.Fatal("AllDraftsAccepted=true, want false")
	}
	if got.AcceptedPrefixLen != 2 || got.VerifiedCount != 2 {
		t.Fatalf("accepted/verified=%d/%d, want 2/2", got.AcceptedPrefixLen, got.VerifiedCount)
	}
	if got.FirstRejectedIndex != 2 {
		t.Fatalf("FirstRejectedIndex=%d, want 2", got.FirstRejectedIndex)
	}
	if got.BonusToken != 42 {
		t.Fatalf("BonusToken=%d, want verifier mismatch 42", got.BonusToken)
	}
	if !sameInts(got.AcceptedTokens, []int{10, 11}) || !sameInts(got.OutputTokens, []int{10, 11, 42}) {
		t.Fatalf("accepted=%v output=%v", got.AcceptedTokens, got.OutputTokens)
	}
}

func TestAcceptMTPDraftFirstTokenMismatch(t *testing.T) {
	got, err := AcceptMTPDraft([]int{10, 11}, []int{77, 88, 99})
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	if got.AcceptedPrefixLen != 0 || got.BonusToken != 77 || got.FirstRejectedIndex != 0 {
		t.Fatalf("got accepted=%d bonus=%d rejected=%d, want 0/77/0", got.AcceptedPrefixLen, got.BonusToken, got.FirstRejectedIndex)
	}
	if !sameInts(got.OutputTokens, []int{77}) {
		t.Fatalf("output=%v, want [77]", got.OutputTokens)
	}
}

func TestAcceptMTPDraftNoDraftsEmitsVerifierToken(t *testing.T) {
	got, err := AcceptMTPDraft(nil, []int{123})
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	if !got.AllDraftsAccepted || got.AcceptedPrefixLen != 0 || got.BonusToken != 123 {
		t.Fatalf("got accepted=%d all=%v bonus=%d, want 0/true/123", got.AcceptedPrefixLen, got.AllDraftsAccepted, got.BonusToken)
	}
	if !sameInts(got.OutputTokens, []int{123}) {
		t.Fatalf("output=%v, want [123]", got.OutputTokens)
	}
}

func TestMTPAcceptanceKVKeepTokens(t *testing.T) {
	got := MTPAcceptance{AcceptedPrefixLen: 3}.KVKeepTokens()
	if got != 4 {
		t.Fatalf("KVKeepTokens=%d want 4", got)
	}
}

func TestCommitAcceptedFloatKV(t *testing.T) {
	acceptance := MTPAcceptance{AcceptedPrefixLen: 1} // keep accepted token + bonus = 2 staged positions
	k := [][]float32{{1, 2, 10, 11, 12, 13, 14, 15}}
	v := [][]float32{{3, 4, 20, 21, 22, 23, 24, 25}}
	cp := FloatKVCheckpoint{KLen: []int{2}, VLen: []int{2}}
	if err := CommitAcceptedFloatKV(k, v, cp, []int{2}, acceptance); err != nil {
		t.Fatalf("CommitAcceptedFloatKV: %v", err)
	}
	if want := []float32{1, 2, 10, 11, 12, 13}; !sameFloat32s(k[0], want) {
		t.Fatalf("K=%v want %v", k[0], want)
	}
	if want := []float32{3, 4, 20, 21, 22, 23}; !sameFloat32s(v[0], want) {
		t.Fatalf("V=%v want %v", v[0], want)
	}
}

func TestCommitAcceptedCompressedKV(t *testing.T) {
	cache := NewCompressedKVCache(2, 1, 2, nil, true)
	cache.Append([]float32{1, 2}, []float32{10, 20})
	cp := CheckpointCompressedKV([]*CompressedKVCache{cache})
	cache.Append([]float32{3, 4}, []float32{30, 40})
	cache.Append([]float32{5, 6}, []float32{50, 60})
	cache.Append([]float32{7, 8}, []float32{70, 80})
	acceptance := MTPAcceptance{AcceptedPrefixLen: 1} // keep two staged positions
	if err := CommitAcceptedCompressedKV([]*CompressedKVCache{cache}, cp, acceptance); err != nil {
		t.Fatalf("CommitAcceptedCompressedKV: %v", err)
	}
	if got, want := cache.SeqLen(), 3; got != want {
		t.Fatalf("seq len=%d want %d", got, want)
	}
	if want := []float32{1, 2, 3, 4, 5, 6}; !sameFloat32s(cache.GetK(), want) {
		t.Fatalf("K=%v want %v", cache.GetK(), want)
	}
}

func TestAcceptMTPDraftRejectsWrongVerifierCount(t *testing.T) {
	if _, err := AcceptMTPDraft([]int{1, 2}, []int{1, 2}); err == nil {
		t.Fatal("AcceptMTPDraft accepted verifier count without bonus token")
	}
	if _, err := AcceptMTPDraft([]int{1, 2}, []int{1, 2, 3, 4}); err == nil {
		t.Fatal("AcceptMTPDraft accepted oversized verifier count")
	}
}

func TestAcceptMTPDraftFromLogits(t *testing.T) {
	got, err := AcceptMTPDraftFromLogits([]int{1, 2, 3}, [][]float32{
		{0, 9, 1, 0}, // verifier token 1 accepts draft[0]
		{0, 1, 8, 0}, // verifier token 2 accepts draft[1]
		{0, 0, 1, 7}, // verifier token 3 accepts draft[2]
		{0, 4, 1, 0}, // bonus token 1
	})
	if err != nil {
		t.Fatalf("AcceptMTPDraftFromLogits: %v", err)
	}
	if !got.AllDraftsAccepted || got.BonusToken != 1 || !sameInts(got.OutputTokens, []int{1, 2, 3, 1}) {
		t.Fatalf("got all=%v bonus=%d output=%v", got.AllDraftsAccepted, got.BonusToken, got.OutputTokens)
	}
}

func TestAcceptMTPDraftFromLogitsPropagatesErrors(t *testing.T) {
	if _, err := AcceptMTPDraftFromLogits([]int{1}, [][]float32{{0, 1}}); err == nil {
		t.Fatal("AcceptMTPDraftFromLogits accepted missing bonus row")
	}
	if _, err := AcceptMTPDraftFromLogits([]int{1}, [][]float32{{0, 1}, nil}); err == nil {
		t.Fatal("AcceptMTPDraftFromLogits accepted empty verifier logits row")
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
