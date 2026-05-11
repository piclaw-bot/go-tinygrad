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
