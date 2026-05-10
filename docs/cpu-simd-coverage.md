# CPU SIMD Coverage Audit

This is the baseline for Phase 4b: making CPU inference hit AVX2/FMA or NEON
wrappers for every hot decode/prefill primitive, with scalar Go as fallback.

## Current coverage map

| Hot path | Current implementation | AVX2/FMA | NEON | Notes / next action |
|---|---|---:|---:|---|
| `RMSNorm` F32 | `simd.RMSNorm` wrapper | ✅ | ✅ | Used by decoder and `ForwardLayer` |
| `RMSNormBF16` on F32 buffers | `simd.RMSNormBF16` wrapper | ✅ | ✅* | arm64 runtime verification pending |
| `RMSNormNoScale` | `simd.RMSNormNoScale` wrapper | ✅ | ✅* | Gemma4 V norm still has some scalar call-sites |
| Residual add | `simd.VecAdd` | ✅ | ✅ | Decoder and `ForwardLayer` use wrapper |
| Residual + scale | `simd.VecScaleAdd` | ✅ | ✅ | Available; not yet used everywhere |
| `ToBF16` | `simd.ToBF16` | ✅ | ✅ | Used for Gemma3/4 truncation semantics |
| SiLU × Mul | `simd.VecSiLUMul` wrapper | wrapper only | wrapper only | Currently jumps to Go due `exp`; candidate for polynomial SIMD approximation |
| GELU(tanh) × Mul | `simd.GELUTanhMul` wrapper | wrapper only | wrapper only | Centralized in decoder, `ForwardLayer`, PLI fallback |
| RoPE | scalar Go | ❌ | ❌ | Needs vectorized pair rotation |
| RoPEPartial | scalar Go | ❌ | ❌ | High priority for Gemma4 CPU path |
| GQA attention scores | `simd.Sdot` per head/token | ✅ | ✅ | Intermediate improvement; still allocates scores per head |
| GQA attention output | scalar Go | ❌ | ❌ | Candidate for fused per-head decode kernel |
| F32 GEMV dense | `simd.SgemmNN` when pre-transposed | ✅ | ✅ | `gemvNT` path uses `simd.Sdot` row-wise |
| MLX4 GEMV | scalar unpack/dequant loop | ❌ | ❌ | Biggest CPU gap for quantized models and MoE experts |
| GPTQ Q4 GEMV | scalar unpack/dequant loop | ❌ | ❌ | Needs AVX2/NEON nibble unpack + FMA |
| MoE CPU experts | parallel goroutines + MLX4 scalar GEMV | partial | partial | Activation now goes through SIMD wrapper; GEMV dominates |
| BERT/GTE encoder | workspace + SGEMM/SIMD vec ops | ✅ | ✅ | Already comparatively mature |
| TurboQuant rotation/dequant | scalar matvec + bit unpack | ❌ | ❌ | Needs scratch reuse and SIMD matvec/unpack |

`✅*` means the assembly exists/cross-compiles but still needs runtime proof on arm64 hardware.

## Benchmarks added

`model/cpu_hotpath_bench_test.go` adds synthetic baselines for:

- `BenchmarkCPUHotRMSNorm3584`
- `BenchmarkCPUHotGELUTanhMul8192`
- `BenchmarkCPUHotSiLUMul8192`
- `BenchmarkCPUHotRoPEPartialGemma4SWA`
- `BenchmarkCPUHotGQAAttentionDecode512`
- `BenchmarkCPUHotGemvMLQ1536x2048`

Run with:

```bash
go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```

## Baseline snapshot (i7-12700, amd64)

```text
BenchmarkCPUHotRMSNorm3584              ~0.50 µs/op, 0 allocs
BenchmarkCPUHotGELUTanhMul8192          ~187 µs/op, 0 allocs
BenchmarkCPUHotSiLUMul8192              ~69 µs/op, 0 allocs
BenchmarkCPUHotRoPEPartialGemma4SWA     ~2.6 µs/op, 0 allocs
BenchmarkCPUHotGQAAttentionDecode512    ~1.1 ms/op, 13 allocs  (after Sdot wiring)
BenchmarkCPUHotGemvMLQ1536x2048         ~10.4 ms/op, 0 allocs
```

## Immediate next steps

1. Vectorize `RoPEPartial` on AVX2 and NEON.
2. Add MLX4 GEMV SIMD kernels for CPU quantized decode and MoE experts.
3. Eliminate GQA attention scratch allocations and fuse score/softmax/output where practical.
4. Add allocation gates for decode once scratch-buffer reuse lands.
5. Runtime-verify arm64 NEON kernels on Orange Pi 6+.
