package gpu

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/rcarmo/go-pherence/backends/placement"
)

// ExpertPool manages a fixed number of MoE expert weight sets on GPU.
// Experts are cached with LRU eviction and hit/miss tracking.
type ExpertPool struct {
	mu     sync.Mutex
	slots  int                      // max experts on GPU simultaneously
	cache  map[int]*ExpertEntry     // expert_id → cached entry
	order  []int                    // LRU order (most recent at end)
	budget *placement.BudgetManager // optional budget tracking

	// Stats
	Hits   atomic.Uint64
	Misses atomic.Uint64
	Evicts atomic.Uint64
}

// ExpertEntry holds one expert's GPU-resident weights.
type ExpertEntry struct {
	ExpertID  int
	GateW     *GPUMLXWeight // gate projection [hidden → moe_inter]
	UpW       *GPUMLXWeight // up projection [hidden → moe_inter]
	DownW     *GPUMLXWeight // down projection [moe_inter → hidden]
	SizeBytes int64         // total VRAM used
}

// NewExpertPool creates a pool with the given number of GPU slots.
func NewExpertPool(slots int, budget *placement.BudgetManager) *ExpertPool {
	return &ExpertPool{
		slots:  slots,
		cache:  make(map[int]*ExpertEntry),
		budget: budget,
	}
}

// Peek returns the cached expert without changing hit/miss/eviction stats or
// LRU order. Use it for planning checks; use Get for actual expert use.
func (p *ExpertPool) Peek(expertID int) *ExpertEntry {
	if p == nil || expertID < 0 {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.cache[expertID]
}

// Get returns the cached expert, or nil if not present (miss).
// On hit, the expert is moved to the most-recently-used position.
func (p *ExpertPool) Get(expertID int) *ExpertEntry {
	if p == nil || expertID < 0 {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	entry, ok := p.cache[expertID]
	if ok {
		p.touchLocked(expertID)
		p.Hits.Add(1)
		if p.budget != nil {
			p.budget.Hit(placement.BudgetExpert)
		}
		return entry
	}
	p.Misses.Add(1)
	return nil
}

// Put adds an expert to the pool, evicting the LRU entry if full.
// Returns the evicted entry (if any) so the caller can free its GPU resources.
func (p *ExpertPool) Put(entry *ExpertEntry) *ExpertEntry {
	if entry == nil {
		return nil
	}
	if p == nil || entry.ExpertID < 0 {
		return entry
	}
	if p.slots <= 0 {
		// A zero-slot pool is disabled. Return the entry so callers that just
		// uploaded GPU resources can release it via FreeExpertEntry.
		return entry
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// Already cached? Replace the old GPU resources and return them for release.
	if old, ok := p.cache[entry.ExpertID]; ok {
		p.cache[entry.ExpertID] = entry
		p.touchLocked(entry.ExpertID)
		if p.budget != nil {
			p.budget.Free(placement.BudgetExpert, old.SizeBytes)
			p.budget.Alloc(placement.BudgetExpert, entry.SizeBytes)
		}
		return old
	}

	var evicted *ExpertEntry
	// Evict if full.
	if len(p.cache) >= p.slots {
		evicted = p.evictLRULocked()
	}

	p.cache[entry.ExpertID] = entry
	p.order = append(p.order, entry.ExpertID)

	if p.budget != nil {
		p.budget.Alloc(placement.BudgetExpert, entry.SizeBytes)
	}

	return evicted
}

// EvictLRU explicitly evicts the least-recently-used expert.
// Returns the evicted entry or nil if pool is empty.
func (p *ExpertPool) EvictLRU() *ExpertEntry {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.evictLRULocked()
}

func (p *ExpertPool) evictLRULocked() *ExpertEntry {
	if p == nil || len(p.order) == 0 {
		return nil
	}
	lruID := p.order[0]
	p.order = p.order[1:]
	entry, ok := p.cache[lruID]
	if ok {
		delete(p.cache, lruID)
		p.Evicts.Add(1)
		if p.budget != nil {
			p.budget.Free(placement.BudgetExpert, entry.SizeBytes)
			p.budget.Evict(placement.BudgetExpert)
		}
	}
	return entry
}

func (p *ExpertPool) touchLocked(expertID int) {
	if p == nil || expertID < 0 {
		return
	}
	// Move to end of LRU order
	for i, id := range p.order {
		if id == expertID {
			p.order = append(p.order[:i], p.order[i+1:]...)
			break
		}
	}
	p.order = append(p.order, expertID)
}

// Size returns the number of currently cached experts.
func (p *ExpertPool) Size() int {
	if p == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.cache)
}

// Slots returns the maximum number of expert slots.
func (p *ExpertPool) Slots() int {
	if p == nil {
		return 0
	}
	return p.slots
}

// Report returns a human-readable summary.
func (p *ExpertPool) Report() string {
	if p == nil {
		return "experts: 0/0 cached  hits=0 misses=0 evicts=0"
	}
	p.mu.Lock()
	n := len(p.cache)
	p.mu.Unlock()
	return fmt.Sprintf("experts: %d/%d cached  hits=%d misses=%d evicts=%d",
		n, p.slots,
		p.Hits.Load(), p.Misses.Load(), p.Evicts.Load())
}

// FreeExpertEntry releases GPU resources for an evicted expert.
func FreeExpertEntry(e *ExpertEntry) {
	if e == nil {
		return
	}
	if e.GateW != nil {
		e.GateW.Free()
	}
	if e.UpW != nil {
		e.UpW.Free()
	}
	if e.DownW != nil {
		e.DownW.Free()
	}
}
