# Performance

## Benchmark Matrix

Hardware: RTX 3060 12GB (sm_86, Ampere) + i7-12700 6-core + 64GB DDR4

| Model | Arch | Format | GPU tok/s | GPU ms/tok | CPU tok/s | CPU ms/tok |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | qwen2 | MLX 4-bit | **~120–158** | ~6–8 | 1.1 | 912 |
| SmolLM2-135M | llama | BF16 | **86** | 11.6 | 35.5 | 28 |
| Gemma3-1B | gemma3 | MLX 4-bit | **~72** | ~14 | 4.9 | 203 |
| Qwen2.5-7B | qwen2 | GPTQ 4-bit | **51** | 19.7 | 0.9 | 1060 |
| Qwen2.5-0.5B | qwen2 | MLX 4-bit | **31** | 32.0 | 7.2 | 140 |
| Qwen3-0.6B | qwen3 | MLX 4-bit | **25** | 39.6 | 7.2 | 138 |
| Gemma4-E2B | gemma4 | MLX 4-bit | **~21–22** | ~46–57 | — | — |
| Qwen3-30B MoE | qwen3_moe | MLX 4-bit | **~5.2 cold / ~5.5 warm** | ~180–195 | 0.6 | 1648 |
| Qwen3-0.6B | qwen3 | BF16 | — | — | 7.8 | 129 |
| Gemma3-1B | gemma3 | BF16 | — | — | 4.9 | 203 |

## SIMD Microbenchmarks (3584 elements, i7-12700)

| Operation | F32 AVX2 | BF16 AVX2 | BF16 Go scalar |
|---|---|---|---|
| RMSNorm | **677 ns** | **1,391 ns** | 9,534 ns |
| Dot product | — | **445 ns** | 4,291 ns |
| VecAdd | **190 ns** | SIMD | — |
| ToBF16 | **179 ns** | — | — |
| BF16 Widen→F32 | — | **292 ns** | — |
| BF16 F32→Narrow | — | **147 ns** | — |

## GPU Kernel Performance

| Kernel | Time | Accuracy | Shared mem |
|---|---|---|---|
| SGEMM 16×16 | 348 GFLOPS @ 1024² | — | tiled |
| Q4 GEMV (GPTQ) | ~300µs @ 3584² | 1.7e-6 maxDiff | CUDA tiled + 8× unroll; CPU scalar owner is `runtime/quant` with upfront shape validation |
| Q4 GEMV (MLX) | ~300µs @ 3584² | 6.7e-6 maxDiff | CUDA 8× unroll; CPU scalar owner is `runtime/quant` with dtype/shape validation |
| LM Head GEMV | F32 path for moderate heads, compact MLX path for very large heads | — | 2D grid or quantized MLX GEMV by policy |
| RMSNorm | ~2µs @ 3584 | Newton-refined rsqrt | 256-thread reduce |
| BF16 RMSNorm | ~2µs @ 3584 | native cvt on Ampere+ | 256-thread reduce |

## Where Time Goes (7B decode, single token)

| Phase | GPU (CUDA) | CPU (AVX2) |
|---|---|---|
| 28 transformer layers | ~0.17–0.2s per 16-token short run | ~850 ms/token |
| LM head (152K vocab) | ~1ms total with compact MLX head on short runs | ~60 ms/token (parallel SIMD) |
| Embedding + sampling | ~0.1 ms/token | ~0.1 ms/token |
| **Total** | **~120 tok/s no-profile, up to ~158 tok/s profiled short run** | **~1 tok/s** |

## Stock-weight speculative decoding scaffold

The Orthrus-inspired stock-weight speculative path is currently an **opt-in CPU-only correctness scaffold**, not a performance feature yet:

```bash
go run ./cmd/llmgen -model models/smollm2-135m \
  -prompt "abc abc abc abc" -tokens 8 \
  -speculative -speculative-proposer prompt -speculative-debug
```

Current backend/proposer status:

| Component | Options | Status |
|---|---|---|
| Verifier backend | `replay` | Exact greedy verification by replaying the prepared CPU prompt; useful for parity/acceptance measurement but often slower |
| Future verifier backend | `kv` selector accepted but falls back to `replay` | Planned KV-reusing verifier block |
| Proposers | `prompt`, `repeat-last`, `none` | Stock-weight only; no Orthrus custom diffusion weights |
| Benchmark harness | `cmd/specbench` | Emits CSV with parity, speedup, backend/proposer, acceptance/fallback, emitted tokens, tokens/step, average proposal length, and aggregate workload rows |

Example replay-backend smoke results on `smollm2-135m` are intentionally slower than normal generation, despite exact output parity:

| Prompt | Tokens | Proposer | Backend | Normal tok/s | Spec tok/s | Speedup | Notes |
|---|---:|---|---|---:|---:|---:|---|
| `abc abc abc abc` | 4 | `prompt` | `replay` | ~15.8 | ~5.3 | ~0.33× | prompt lookup gated tiny proposals, so fallback-only |
| `ha ha ha ha` | 4 | `repeat-last` | `replay` | ~15.1 | ~5.3 | ~0.35× | multi-token proposal accepted, but replay verifier dominates |

Use `specbench` for less noisy comparisons:

```bash
go run ./cmd/specbench -model models/smollm2-135m \
  -prompt-file prompts.txt -tokens 16 -repeat 3 \
  -speculative-proposer prompt -csv specbench.csv
```

The expected benefit requires replacing `backend=replay` with a stateful/KV-reusing verifier block. Until then, the useful metrics are acceptance rate, emitted tokens, and tokens/step, not raw speedup.

## NVFP4 / FP4 watchlist

NVFP4 is now a roadmap item rather than an implemented format. Public Hugging
Face searches found relevant checkpoints including `nvidia/Qwen3-8B-NVFP4`,
`NVFP4/Qwen3-32B-FP4`, `nvidia/Qwen3-30B-A3B-NVFP4`,
`nvidia/Gemma-4-31B-IT-NVFP4`, and community Gemma4 26B-A4B NVFP4 artifacts.
See [nvfp4.md](nvfp4.md) for the support track.

Implementation priority should be metadata/detection first, then a
correctness-first CPU dequant path, then CUDA upload/GEMV/GEMM. For Qwen3 MoE,
NVFP4 must be evaluated together with expert-cache/prefetch redesign because the
current bottleneck is cold-miss upload rather than only arithmetic throughput.

## MLX vs GPTQ on GPU

MLX 4-bit is **faster** than GPTQ 4-bit for the same model in the current CUDA path (roughly 120+ vs 51 tok/s for short 7B runs):

- MLX uses `group_size=64` (vs GPTQ's 128) → better cache utilization
- MLX weights transposed to GPTQ layout at upload → reuses fast tiled kernel
- Bias correction kernel adds ~10% overhead but amortized in pipeline

## MoE Performance (Qwen3-30B-A3B)

128 experts per layer, 8 active per token, 48 layers. The current CUDA path keeps the router and selected experts on GPU when possible, uploads cold experts into an LRU expert cache, and accumulates expert outputs on device.

| Configuration | tok/s / time | Notes |
|---|---|---|
| CPU sequential experts | ~0.1 tok/s | baseline |
| CPU parallel experts (8 goroutines) | ~0.6 tok/s | pre-GPU-cache CPU fallback |
| Early GPU attention + CPU experts | ~0.4 tok/s | historical, before GPU expert cache hot path |
| Current GPU router + GPU expert cache, cold route set | ~5.2 tok/s, ~4.0–4.5s for 16 tokens | cold route set uploads selected experts and uses them immediately |
| Current GPU router + GPU expert cache, warm route set | ~5.5 tok/s, ~2.9s for 16 tokens | zero expert misses on repeated prompt |

Per-expert VRAM is about 2.3 MB for Qwen3-30B-A3B MLX4 gate/up/down projections. Expert pool capacity is about 4072 slots with an F32 LM head resident on the 12GB RTX 3060 test system. Warm-run `GO_PHERENCE_PROFILE_DECODE=1` counters are roughly `kernels=123680 h2d=44 d2h=1388 d2d=6720 syncs=32` for a 16-token repeat, so the next major bottleneck is launch/copy count rather than CPU expert GEMV.
