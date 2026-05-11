# Weight Budget Manager — Design

## Motivation

Models are getting larger than any single memory tier. A 7B MLX4 model is ~4GB
of weights, but a 70B MoE model might be 40GB+ with 128 experts. We need to
manage weights across tiers with explicit budgets, not just "load everything."

Inspired by the [ds4 streaming PR](https://github.com/antirez/ds4/pull/24),
which adds mmap-backed streamed weight access with hot-residency plans,
madvise-based eviction, and expert cache tracking for DeepSeek on Metal.

## Eager mmap loading

By default, safetensors files are mmap'd lazily, so the OS may fault pages during
first use. For batch/server deployments where predictable first-token latency is
more important than fastest startup, pass `--eager-load` (or set
`GO_PHERENCE_EAGER_LOAD=1`). This issues `MADV_WILLNEED` for each mmap'd shard
and touches one byte per page at model load time.

```bash
./bin/llmserver -model models/qwen2.5-7b-mlx4 --eager-load -gpu
./bin/llmgen -model models/gemma4-e2b-mlx4 --eager-load --turbo-quant
```

The loader logs the total mapped bytes and elapsed pre-fault time. Sharded models
pre-fault each shard and report the aggregate size.

## Memory Tiers

```
Tier 0: GPU VRAM          fastest, smallest (12GB RTX 3060)
Tier 1: Pinned CPU RAM    fast DMA to GPU, limited by system RAM
Tier 2: Regular heap      Go-managed, GC pressure
Tier 3: mmap (disk)       OS page cache, madvise control
```

## Implementation status

Backend-neutral budget and layer-placement policy now lives in `backends/placement`:

- `BudgetManager` tracks resident/layer/stream/expert budgets and hit/evict counters, with guarded accounting for negative/overflowing inputs.
- `PlanLayerPlacement` estimates per-layer/resident weight sizes from model dimensions and accepts caller-supplied device-memory availability, keeping policy independent from CUDA/Vulkan discovery; invalid dimensions are clamped for safe planning.
- `runtime/memory.MmapAdvisor` tracks mmap residency ranges and madvise hints with idempotent hot-byte accounting; `loader/safetensors` uses it for eager pre-faulting and future streamed weight access.
- GPU-resident expert cache entries remain in `gpu` because they own `GPUMLXWeight` device resources, but they use `backends/placement.BudgetManager` for accounting and handle disabled/replacement cases explicitly.

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

func (a *MmapAdvisor) Prefetch(offset, bytes int64) error // MADV_WILLNEED
func (a *MmapAdvisor) Evict(offset, bytes int64) error     // MADV_DONTNEED
func (a *MmapAdvisor) Touch(offset, bytes int64)            // update tracking
```

### Phase 3: Layer Placement Policy

```go
type LayerPlacement struct {
    Layer      int
    Location   Tier      // GPU, PinnedCPU, Mmap
    WeightSize int64     // bytes on this tier
    KVSize     int64     // KV cache bytes
}

func PlanLayerPlacement(info placement.ModelSizeInfo, gpuLayers int, availGPUBytes uint64) placement.PlacementPlan
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
| `hot_residency_plan` | `backends/placement.PlanLayerPlacement()` with resident estimates |
| `madvise(DONTNEED/WILLNEED)` | `runtime/memory.MmapAdvisor.Evict()/Prefetch()` |
| `g_model_stream_hit/evict_count` | `backends/placement.BudgetManager` hit/evict counters |
| `compact_expert_cache` | `ExpertPool` with LRU |
| Metal shared memory | CUDA pinned memory + explicit DMA |
| `split_after_layers=1` (streaming) | Layer-at-a-time GPU forward |
| `DS4_METAL_RESIDENT_HOT_MB` | `--resident-mb` flag |
| `hot_plan_add_tensor` | `backends/placement.BudgetManager.Alloc(...)` |
| `hot_plan_merge` (range merging) | `runtime/memory.MmapAdvisor` range coalescing |

The key difference: ds4 runs on unified memory (Metal shared), so madvise
directly controls GPU-visible residency. go-pherence runs on discrete GPU,
so we need explicit DMA staging — but the budget/tracking/eviction logic
is the same, just with an extra copy step.
