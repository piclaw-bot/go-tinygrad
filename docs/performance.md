# Performance

## Benchmark Matrix

Hardware: RTX 3060 12GB (sm_86, Ampere) + i7-12700 6-core + 64GB DDR4

| Model | Arch | Format | GPU tok/s | GPU ms/tok | CPU tok/s | CPU ms/tok |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | qwen2 | MLX 4-bit | **217** | 4.6 | 1.1 | 912 |
| SmolLM2-135M | llama | BF16 | **86** | 11.6 | 35.5 | 28 |
| Qwen2.5-7B | qwen2 | GPTQ 4-bit | **51** | 19.7 | 0.9 | 1060 |
| Qwen2.5-0.5B | qwen2 | MLX 4-bit | **31** | 32.0 | 7.2 | 140 |
| Qwen3-0.6B | qwen3 | MLX 4-bit | **25** | 39.6 | 7.2 | 138 |
| Gemma3-1B | gemma3 | MLX 4-bit | **18** | 55.4 | 4.9 | 203 |
| Gemma4-E2B | gemma4 | MLX 4-bit | **14** | 74.1 | — | — |
| Qwen3-30B MoE | qwen3_moe | MLX 4-bit | **0.4** | 2519 | 0.6 | 1648 |
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
| Q4 GEMV (GPTQ) | ~300µs @ 3584² | 1.7e-6 maxDiff | tiled + 8× unroll |
| Q4 GEMV (MLX) | ~300µs @ 3584² | 6.7e-6 maxDiff | 8× unroll |
| LM Head GEMV | ~5ms @ 152K×3584 | — | 2D grid |
| RMSNorm | ~2µs @ 3584 | Newton-refined rsqrt | 256-thread reduce |
| BF16 RMSNorm | ~2µs @ 3584 | native cvt on Ampere+ | 256-thread reduce |

## Where Time Goes (7B decode, single token)

| Phase | GPU (CUDA) | CPU (AVX2) |
|---|---|---|
| 28 transformer layers | ~2 ms | ~850 ms |
| LM head (152K vocab) | ~3 ms (GPU kernel) | ~60 ms (parallel SIMD) |
| Embedding + sampling | ~0.1 ms | ~0.1 ms |
| **Total** | **~5 ms** | **~910 ms** |

## MLX vs GPTQ on GPU

MLX 4-bit is **faster** than GPTQ 4-bit for the same model (217 vs 51 tok/s for 7B):

- MLX uses `group_size=64` (vs GPTQ's 128) → better cache utilization
- MLX weights transposed to GPTQ layout at upload → reuses fast tiled kernel
- Bias correction kernel adds ~10% overhead but amortized in pipeline

## MoE Performance (Qwen3-30B-A3B)

128 experts per layer, 8 active per token, 48 layers.

| Configuration | tok/s | Notes |
|---|---|---|
| CPU sequential experts | 0.1 | baseline |
| CPU parallel experts (8 goroutines) | **0.6** | 6× from parallelism |
| GPU attention + CPU experts | **0.4** | attention on GPU, experts on CPU |
| GPU attention + GPU expert cache | 0.2 | cold-miss upload dominates (correct output) |

Per-expert VRAM: 2.3 KB (MLX4). Expert pool: 4076 slots in 9.2 GB.
Bottleneck is CPU expert MLP (384 GEMVs per token × 48 layers).
