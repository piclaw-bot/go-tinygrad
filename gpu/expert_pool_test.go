package gpu

import (
	"testing"

	"github.com/rcarmo/go-pherence/backends/placement"
)

func TestExpertPoolBasic(t *testing.T) {
	pool := NewExpertPool(3, nil)

	// Insert 3 experts
	for i := 0; i < 3; i++ {
		evicted := pool.Put(&ExpertEntry{ExpertID: i, SizeBytes: 1024 * 1024})
		if evicted != nil {
			t.Fatalf("should not evict when pool not full (expert %d)", i)
		}
	}
	if pool.Size() != 3 {
		t.Fatalf("expected 3 cached, got %d", pool.Size())
	}

	// Hit expert 0
	e := pool.Get(0)
	if e == nil || e.ExpertID != 0 {
		t.Fatal("expected hit on expert 0")
	}
	if pool.Hits.Load() != 1 {
		t.Fatalf("expected 1 hit, got %d", pool.Hits.Load())
	}

	// Miss on expert 5
	e = pool.Get(5)
	if e != nil {
		t.Fatal("expected miss on expert 5")
	}
	if pool.Misses.Load() != 1 {
		t.Fatalf("expected 1 miss, got %d", pool.Misses.Load())
	}

	// Insert expert 5 — should evict LRU (expert 1, since 0 was touched)
	evicted := pool.Put(&ExpertEntry{ExpertID: 5, SizeBytes: 1024 * 1024})
	if evicted == nil {
		t.Fatal("expected eviction")
	}
	if evicted.ExpertID != 1 {
		t.Fatalf("expected LRU eviction of expert 1, got %d", evicted.ExpertID)
	}
	if pool.Evicts.Load() != 1 {
		t.Fatalf("expected 1 evict, got %d", pool.Evicts.Load())
	}

	// Expert 1 should be gone, expert 0 should still be cached
	if pool.Get(1) != nil {
		t.Fatal("expert 1 should be evicted")
	}
	if pool.Get(0) == nil {
		t.Fatal("expert 0 should still be cached")
	}

	t.Log(pool.Report())
}

func TestExpertPoolPeekDoesNotMutateStatsOrLRU(t *testing.T) {
	pool := NewExpertPool(2, nil)
	pool.Put(&ExpertEntry{ExpertID: 0, SizeBytes: 100})
	pool.Put(&ExpertEntry{ExpertID: 1, SizeBytes: 100})

	if got := pool.Peek(0); got == nil || got.ExpertID != 0 {
		t.Fatalf("Peek(0)=%#v, want expert 0", got)
	}
	if got := pool.Peek(99); got != nil {
		t.Fatalf("Peek missing expert=%#v, want nil", got)
	}
	if pool.Hits.Load() != 0 || pool.Misses.Load() != 0 || pool.Evicts.Load() != 0 {
		t.Fatalf("Peek mutated stats: hits=%d misses=%d evicts=%d", pool.Hits.Load(), pool.Misses.Load(), pool.Evicts.Load())
	}

	// If Peek updated LRU order, expert 1 would be evicted here. It must not.
	evicted := pool.Put(&ExpertEntry{ExpertID: 2, SizeBytes: 100})
	if evicted == nil || evicted.ExpertID != 0 {
		id := -1
		if evicted != nil {
			id = evicted.ExpertID
		}
		t.Fatalf("Peek should not update LRU; evicted %d, want 0", id)
	}
}

func TestExpertPoolWithBudget(t *testing.T) {
	budget := placement.NewBudgetManager(0, 0, 0, 10) // 10MB expert budget
	pool := NewExpertPool(5, budget)

	// Fill with experts
	for i := 0; i < 5; i++ {
		pool.Put(&ExpertEntry{ExpertID: i, SizeBytes: 2 * 1024 * 1024}) // 2MB each
	}

	// Budget should show 10MB used
	if budget.ExpertUsed != 10*1024*1024 {
		t.Fatalf("expected 10MB used, got %d", budget.ExpertUsed)
	}

	// Evict one
	evicted := pool.Put(&ExpertEntry{ExpertID: 10, SizeBytes: 2 * 1024 * 1024})
	if evicted == nil {
		t.Fatal("expected eviction")
	}
	FreeExpertEntry(evicted)

	// Budget should still show 10MB (evicted 2MB, added 2MB)
	if budget.ExpertUsed != 10*1024*1024 {
		t.Fatalf("expected 10MB used after swap, got %d", budget.ExpertUsed)
	}

	t.Log(pool.Report())
	t.Log(budget.Report())
}

func TestExpertPoolDisabled(t *testing.T) {
	pool := NewExpertPool(0, nil)
	entry := &ExpertEntry{ExpertID: 1, SizeBytes: 100}
	if evicted := pool.Put(entry); evicted != entry {
		t.Fatalf("disabled pool should return inserted entry for release, got %#v", evicted)
	}
	if pool.Size() != 0 {
		t.Fatalf("disabled pool cached %d entries", pool.Size())
	}
	if pool.Get(1) != nil {
		t.Fatal("disabled pool should not return entries")
	}
	if evicted := pool.Put(nil); evicted != nil {
		t.Fatalf("nil entry should be ignored, got %#v", evicted)
	}
}

func TestExpertPoolReplaceReturnsOldEntry(t *testing.T) {
	budget := placement.NewBudgetManager(0, 0, 0, 10)
	pool := NewExpertPool(2, budget)
	old := &ExpertEntry{ExpertID: 7, SizeBytes: 2 * 1024 * 1024}
	if evicted := pool.Put(old); evicted != nil {
		t.Fatalf("initial insert evicted %#v", evicted)
	}
	newEntry := &ExpertEntry{ExpertID: 7, SizeBytes: 3 * 1024 * 1024}
	if evicted := pool.Put(newEntry); evicted != old {
		t.Fatalf("replacement should return old entry, got %#v", evicted)
	}
	if got := pool.Get(7); got != newEntry {
		t.Fatalf("replacement not cached: %#v", got)
	}
	if budget.ExpertUsed != 3*1024*1024 {
		t.Fatalf("budget after replacement=%d, want 3MB", budget.ExpertUsed)
	}
}

func TestExpertPoolLRUOrder(t *testing.T) {
	pool := NewExpertPool(3, nil)

	// Insert 0, 1, 2
	pool.Put(&ExpertEntry{ExpertID: 0, SizeBytes: 100})
	pool.Put(&ExpertEntry{ExpertID: 1, SizeBytes: 100})
	pool.Put(&ExpertEntry{ExpertID: 2, SizeBytes: 100})

	// Touch 0 and 1 (making 2 the LRU)
	pool.Get(0)
	pool.Get(1)

	// Insert 3 — should evict 2
	evicted := pool.Put(&ExpertEntry{ExpertID: 3, SizeBytes: 100})
	if evicted == nil || evicted.ExpertID != 2 {
		id := -1
		if evicted != nil {
			id = evicted.ExpertID
		}
		t.Fatalf("expected LRU eviction of expert 2, got %d", id)
	}

	// Insert 4 — should evict 0 (oldest after 2 was evicted)
	evicted = pool.Put(&ExpertEntry{ExpertID: 4, SizeBytes: 100})
	if evicted == nil || evicted.ExpertID != 0 {
		id := -1
		if evicted != nil {
			id = evicted.ExpertID
		}
		t.Fatalf("expected LRU eviction of expert 0, got %d", id)
	}
}

func TestExpertPoolNilAndInvalidExpertSafety(t *testing.T) {
	var pool *ExpertPool
	if got := pool.Get(1); got != nil {
		t.Fatalf("nil pool Get=%#v, want nil", got)
	}
	entry := &ExpertEntry{ExpertID: 1, SizeBytes: 10}
	if got := pool.Put(entry); got != entry {
		t.Fatalf("nil pool Put should return entry for release, got %#v", got)
	}
	if got := pool.EvictLRU(); got != nil {
		t.Fatalf("nil pool EvictLRU=%#v, want nil", got)
	}
	if got := pool.Size(); got != 0 {
		t.Fatalf("nil pool Size=%d, want 0", got)
	}
	if got := pool.Slots(); got != 0 {
		t.Fatalf("nil pool Slots=%d, want 0", got)
	}
	if got := pool.Report(); got == "" {
		t.Fatal("nil pool Report should be non-empty")
	}

	pool = NewExpertPool(2, nil)
	bad := &ExpertEntry{ExpertID: -1, SizeBytes: 10}
	if got := pool.Put(bad); got != bad {
		t.Fatalf("invalid expert Put should return entry for release, got %#v", got)
	}
	if pool.Get(-1) != nil {
		t.Fatal("negative expert ID should not hit")
	}
	if pool.Size() != 0 {
		t.Fatalf("invalid expert was cached, size=%d", pool.Size())
	}
}
