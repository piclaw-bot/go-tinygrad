# Weight Budget Manager — Design

## Motivation

Models are getting larger than any single memory tier. A 7B MLX4 model is ~4GB
of weights, but a 70B MoE model might be 40GB+ with 128 experts. We need to
manage weights across tiers with explicit budgets, not just "load everything."

Inspired by the [ds4 streaming PR](https://github.com/antirez/ds4/pull/24),
which adds mmap-backed streamed weight access with hot-residency plans,
madvise-based eviction, and expert cache tracking for DeepSeek on Metal.

## Memory Tiers

```
Tier 0: GPU VRAM          fastest, smallest (12GB RTX 3060)
Tier 1: Pinned CPU RAM    fast DMA to GPU, limited by system RAM
Tier 2: Regular heap      Go-managed, GC pressure
Tier 3: mmap (disk)       OS page cache, madvise control
```

## Budget Categories

### 1. Permanent Residency Budget (`--resident-mb`)

Weights that **never move** once placed:

- **GPU permanent**: embeddings, LM head, norm weights, RoPE tables, KV cache
- **CPU permanent**: tokenizer, config, small tensors

These are loaded once and pinned. The budget caps how much GPU VRAM is
reserved for always-resident weights vs layer/expert pools.

### 2. Layer Budget (`--layer-budget-mb` or `--gpu-layers N`)

Transformer layer weights that cycle through GPU:

- **Full residency**: all N layers on GPU (current behavior, fastest)
- **Partial**: first K layers on GPU, rest on CPU (layer splitting)
- **Streaming**: 1 layer on GPU at a time, weights uploaded per-token
  (slowest but works for any model size)

The layer budget controls how many layers fit in GPU VRAM simultaneously.
Layers not on GPU live in pinned CPU RAM (Tier 1) or mmap (Tier 3).

### 3. Stream/Cache Budget (`--stream-mb`)

Working set for streamed weights — the mmap pages actively touched:

- Controls `madvise(MADV_WILLNEED)` prefetch for next layer's weights
- Controls `madvise(MADV_DONTNEED)` eviction for used layer's weights
- Tracks hit/evict counts per range for budget tuning
- On CUDA: maps to `cuMemcpyHtoDAsync` from pinned staging buffer

This is the ds4 "hot residency plan" concept, generalized:
- ds4 uses Metal shared memory (unified) — madvise controls residency
- go-pherence uses discrete GPU — explicit DMA replaces madvise
- But the budget/tracking/eviction logic is the same

### 4. Expert Budget (`--expert-mb` or `--expert-slots N`)

For MoE models (Mixtral, DeepSeek-V3, Qwen3-MoE):

- **Hot experts**: K expert weight sets on GPU (LRU or frequency-based)
- **Warm experts**: in pinned CPU RAM, ready for fast DMA
- **Cold experts**: mmap-backed, evicted with DONTNEED

Router selects 2-8 experts per token. The expert budget controls how
many experts stay on GPU between tokens. Cold experts need upload time.

## Implementation Plan

### Phase 1: Budget Manager Core

```go
type BudgetManager struct {
    // Configured budgets (bytes)
    ResidentBudget  int64  // GPU permanent (embeddings, LM head, norms)
    LayerBudget     int64  // GPU layer pool
    StreamBudget    int64  // mmap working set / staging buffer
    ExpertBudget    int64  // GPU expert pool (MoE only)

    // Tracking
    ResidentUsed    int64
    LayerUsed       int64
    StreamUsed      int64
    ExpertUsed      int64

    // Eviction stats (for budget tuning)
    LayerHits       uint64
    LayerEvicts     uint64
    ExpertHits      uint64
    ExpertEvicts    uint64
    StreamAdvised   uint64
    StreamEvicted   uint64
}
```

### Phase 2: mmap Advisor

```go
type MmapAdvisor struct {
    base     uintptr  // mmap base address
    size     int64    // total mapped size

    // Per-range tracking (page-aligned)
    ranges   []AdvisedRange
    pending  []AdvisedRange  // WILLNEED queue
}

type AdvisedRange struct {
    Offset   int64
    Bytes    int64
    State    RangeState  // Cold, Prefetching, Hot, Evicting
    Hits     uint64
    LastUsed int64       // timestamp
}

func (a *MmapAdvisor) Prefetch(offset, bytes int64)  // MADV_WILLNEED
func (a *MmapAdvisor) Evict(offset, bytes int64)      // MADV_DONTNEED
func (a *MmapAdvisor) Touch(offset, bytes int64)       // update tracking
```

### Phase 3: Layer Placement Policy

```go
type LayerPlacement struct {
    Layer      int
    Location   Tier      // GPU, PinnedCPU, Mmap
    WeightSize int64     // bytes on this tier
    KVSize     int64     // KV cache bytes
}

func PlanLayerPlacement(model *LlamaConfig, budget *BudgetManager) []LayerPlacement
```

Decision logic:
1. Place permanent tensors (embeddings, LM head, norms) → GPU resident
2. Fill GPU layer pool with layers 0..K-1 until budget exhausted
3. Remaining layers → pinned CPU RAM if available
4. Overflow → mmap streaming

### Phase 4: Layer Streaming Forward Pass

For layers not on GPU:
1. Prefetch next layer weights: `madvise(WILLNEED)` + `cuMemcpyHtoDAsync`
2. Execute current layer on GPU
3. Evict previous layer weights: `madvise(DONTNEED)` + free GPU staging
4. Overlap: prefetch layer N+1 while computing layer N

### Phase 5: Expert Pool (MoE)

```go
type ExpertPool struct {
    Slots     int           // max experts on GPU
    Cache     map[int]*GPUExpertWeights  // expert_id → GPU weights
    LRU       []int         // eviction order
    Hits      uint64
    Misses    uint64
    Evicts    uint64
}

func (p *ExpertPool) Get(expertID int) *GPUExpertWeights  // hit or upload
func (p *ExpertPool) Evict() int                           // free oldest slot
```

## Budget Defaults

| Model class | Resident | Layers | Stream | Experts |
|---|---|---|---|---|
| Small (≤1B) | 200MB | all | 0 | 0 |
| Medium (1-7B) | 500MB | auto-fit | 256MB | 0 |
| Large (7-30B) | 500MB | auto-fit | 512MB | 0 |
| MoE (any) | 500MB | auto-fit | 512MB | 1GB |

## Eviction Tracking Output

```
budget: resident 487/500MB  layers 28/28 (3.2GB)  stream 0MB  experts 0/0
stats:  layer_hits=0  evicts=0  expert_hits=0  evicts=0  stream_advised=0  evicted=0
```

For budget tuning, log per-layer and per-expert hit/evict counts so the
user can adjust `--resident-mb` / `--expert-slots` based on actual usage.

## Relationship to ds4 PR

| ds4 concept | go-pherence equivalent |
|---|---|
| `hot_residency_plan` | `PlanLayerPlacement()` with `ResidentBudget` |
| `madvise(DONTNEED/WILLNEED)` | `MmapAdvisor.Evict()/Prefetch()` |
| `g_model_stream_hit/evict_count` | `BudgetManager.LayerHits/Evicts` |
| `compact_expert_cache` | `ExpertPool` with LRU |
| Metal shared memory | CUDA pinned memory + explicit DMA |
| `split_after_layers=1` (streaming) | Layer-at-a-time GPU forward |
| `DS4_METAL_RESIDENT_HOT_MB` | `--resident-mb` flag |
| `hot_plan_add_tensor` | `BudgetManager.AllocResident()` |
| `hot_plan_merge` (range merging) | `MmapAdvisor` range coalescing |

The key difference: ds4 runs on unified memory (Metal shared), so madvise
directly controls GPU-visible residency. go-pherence runs on discrete GPU,
so we need explicit DMA staging — but the budget/tracking/eviction logic
is the same, just with an extra copy step.
