package placement

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// BudgetCategory represents a memory budget tier.
type BudgetCategory int

const (
	BudgetResident BudgetCategory = iota // GPU permanent (embeddings, LM head, norms)
	BudgetLayer                          // GPU transformer layer pool
	BudgetStream                         // mmap working set / staging buffer
	BudgetExpert                         // GPU MoE expert pool
)

// BudgetManager tracks memory usage and eviction stats across tiers.
type BudgetManager struct {
	mu sync.Mutex

	// Configured budgets (bytes). 0 = unlimited.
	ResidentBudget int64
	LayerBudget    int64
	StreamBudget   int64
	ExpertBudget   int64

	// Current usage (bytes)
	ResidentUsed int64
	LayerUsed    int64
	StreamUsed   int64
	ExpertUsed   int64

	// Eviction counters (atomic for lock-free hot path)
	LayerHits    atomic.Uint64
	LayerEvicts  atomic.Uint64
	ExpertHits   atomic.Uint64
	ExpertEvicts atomic.Uint64
	StreamHits   atomic.Uint64
	StreamEvicts atomic.Uint64
}

// NewBudgetManager creates a manager with the given budgets.
// Pass 0 for any budget to leave it unlimited.
func NewBudgetManager(residentMB, layerMB, streamMB, expertMB int64) *BudgetManager {
	return &BudgetManager{
		ResidentBudget: residentMB * 1024 * 1024,
		LayerBudget:    layerMB * 1024 * 1024,
		StreamBudget:   streamMB * 1024 * 1024,
		ExpertBudget:   expertMB * 1024 * 1024,
	}
}

// NewAutoBudgetManager creates a manager using caller-provided accelerator memory info.
// freeBytes/totalBytes are explicit so this backend-neutral package does not
// depend on a CUDA/Vulkan device query. Allocates: resident first, then layers
// fill remaining, stream/expert from flags.
func NewAutoBudgetManager(freeBytes, totalBytes uint64, residentMB, streamMB, expertMB int64) *BudgetManager {
	if totalBytes == 0 {
		// No accelerator — all budgets are for CPU/mmap only.
		return &BudgetManager{
			StreamBudget: streamMB * 1024 * 1024,
		}
	}

	residentBytes := residentMB * 1024 * 1024
	streamBytes := streamMB * 1024 * 1024
	expertBytes := expertMB * 1024 * 1024

	// Reserve 256MB headroom for work buffers, KV cache growth, etc.
	headroom := int64(256 * 1024 * 1024)
	available := int64(freeBytes) - headroom
	if available < 0 {
		available = 0
	}

	// Resident comes first.
	if residentBytes > available {
		residentBytes = available
	}
	remaining := available - residentBytes

	// Expert pool.
	if expertBytes > remaining {
		expertBytes = remaining
	}
	remaining -= expertBytes

	// Layer pool gets the rest.
	layerBytes := remaining

	return &BudgetManager{
		ResidentBudget: residentBytes,
		LayerBudget:    layerBytes,
		StreamBudget:   streamBytes,
		ExpertBudget:   expertBytes,
	}
}

// Alloc tries to allocate bytes from the given budget category.
// Returns true if the allocation fits, false if it would exceed the budget.
func (b *BudgetManager) Alloc(cat BudgetCategory, bytes int64) bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	budget, used := b.budgetAndUsed(cat)
	if budget > 0 && *used+bytes > budget {
		return false
	}
	*used += bytes
	return true
}

// Free returns bytes to the given budget category.
func (b *BudgetManager) Free(cat BudgetCategory, bytes int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	_, used := b.budgetAndUsed(cat)
	*used -= bytes
	if *used < 0 {
		*used = 0
	}
}

// Available returns remaining bytes in the given budget category.
func (b *BudgetManager) Available(cat BudgetCategory) int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	budget, used := b.budgetAndUsed(cat)
	if budget == 0 {
		return 1<<62 - 1 // unlimited
	}
	avail := budget - *used
	if avail < 0 {
		return 0
	}
	return avail
}

// Hit records a cache hit for the given category.
func (b *BudgetManager) Hit(cat BudgetCategory) {
	switch cat {
	case BudgetLayer:
		b.LayerHits.Add(1)
	case BudgetExpert:
		b.ExpertHits.Add(1)
	case BudgetStream:
		b.StreamHits.Add(1)
	}
}

// Evict records an eviction for the given category.
func (b *BudgetManager) Evict(cat BudgetCategory) {
	switch cat {
	case BudgetLayer:
		b.LayerEvicts.Add(1)
	case BudgetExpert:
		b.ExpertEvicts.Add(1)
	case BudgetStream:
		b.StreamEvicts.Add(1)
	}
}

// Report returns a human-readable budget summary.
func (b *BudgetManager) Report() string {
	b.mu.Lock()
	rBud, rUsed := b.ResidentBudget, b.ResidentUsed
	lBud, lUsed := b.LayerBudget, b.LayerUsed
	sBud, sUsed := b.StreamBudget, b.StreamUsed
	eBud, eUsed := b.ExpertBudget, b.ExpertUsed
	b.mu.Unlock()

	mb := func(v int64) float64 { return float64(v) / (1024 * 1024) }

	return fmt.Sprintf(
		"budget: resident %.0f/%.0fMB  layers %.0f/%.0fMB  stream %.0f/%.0fMB  experts %.0f/%.0fMB\n"+
			"stats:  layer_hits=%d evicts=%d  expert_hits=%d evicts=%d  stream_hits=%d evicts=%d",
		mb(rUsed), mb(rBud), mb(lUsed), mb(lBud), mb(sUsed), mb(sBud), mb(eUsed), mb(eBud),
		b.LayerHits.Load(), b.LayerEvicts.Load(),
		b.ExpertHits.Load(), b.ExpertEvicts.Load(),
		b.StreamHits.Load(), b.StreamEvicts.Load(),
	)
}

func (b *BudgetManager) budgetAndUsed(cat BudgetCategory) (int64, *int64) {
	switch cat {
	case BudgetResident:
		return b.ResidentBudget, &b.ResidentUsed
	case BudgetLayer:
		return b.LayerBudget, &b.LayerUsed
	case BudgetStream:
		return b.StreamBudget, &b.StreamUsed
	case BudgetExpert:
		return b.ExpertBudget, &b.ExpertUsed
	default:
		return 0, &b.ResidentUsed
	}
}
