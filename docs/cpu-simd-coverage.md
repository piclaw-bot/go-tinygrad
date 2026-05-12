# CPU SIMD Coverage Audit

This is the baseline for Phase 4b: making CPU inference hit AVX2/FMA or NEON
wrappers for every hot decode/prefill primitive, with scalar Go as fallback.
The SIMD implementation now lives at import path `github.com/rcarmo/go-pherence/backends/simd` (package name `simd`).

## Current coverage map

| Hot path | Current implementation | AVX2/FMA | NEON | Notes / next action |
|---|---|---:|---:|---|
| `RMSNorm` F32 | `simd.RMSNorm` wrapper | ✅ | ✅ | Used by decoder and `ForwardLayer` |
| `RMSNormBF16` on F32 buffers | `simd.RMSNormBF16` wrapper | ✅ | ✅* | arm64 runtime verification pending |
| `RMSNormNoScale` | `simd.RMSNormNoScale` wrapper | ✅ | ✅* | Gemma4 CPU V norm now routes through wrapper |
| Residual add | `simd.VecAdd` | ✅ | ✅ | Decoder and `ForwardLayer` use wrapper |
| Residual + scale | `simd.VecScaleAdd` / `simd.VecScale` | ✅ | ✅ | Gemma4 layer scalar now uses `VecScale` |
| `ToBF16` | `simd.ToBF16` | ✅ | ✅ | Used for Gemma3/4 truncation semantics |
| SiLU × Mul | `simd.VecSiLUMul` wrapper | wrapper only | wrapper only | Currently jumps to Go due `exp`; candidate for polynomial SIMD approximation |
| GELU(tanh) × Mul | `simd.GELUTanhMul` wrapper | wrapper only | wrapper only | Centralized in decoder, `ForwardLayer`, PLI fallback |
| RoPE | scalar Go | ❌ | ❌ | Needs vectorized pair rotation |
| RoPEPartial | scalar Go | ❌ | ❌ | High priority for Gemma4 CPU path |
| GQA attention scores | `simd.Sdot` per head/token | ✅ | ✅ | Intermediate improvement; still allocates scores per head |
| GQA attention output | `simd.Saxpy` per cached-token V head | ✅ | ✅ | Caller-owned output/score scratch; full fused attention still future work |
| F32 GEMV dense | `simd.SgemmNN` when pre-transposed | ✅ | ✅ | `gemvNT` path uses `simd.Sdot` row-wise |
| MLX4 GEMV | `runtime/quant` scalar unpack/dequant loop with dtype/shape validation | ❌ | ❌ | Biggest CPU gap for quantized models and MoE experts |
| GPTQ Q4 GEMV | `runtime/quant` scalar unpack/dequant loop with qweight/g_idx/scales/qzeros validation | ❌ | ❌ | Needs AVX2/NEON nibble unpack + FMA |
| MoE CPU experts | parallel goroutines + MLX4 scalar GEMV | partial | partial | Activation now goes through SIMD wrapper; GEMV dominates |
| BERT/GTE encoder | workspace + SGEMM/SIMD vec ops | ✅ | ✅ | Already comparatively mature |
| TurboQuant rotation/dequant | scalar matvec + bit unpack | ❌ | ❌ | Needs scratch reuse and SIMD matvec/unpack |

`✅*` means the assembly exists/cross-compiles but still needs runtime proof on arm64 hardware.

## Benchmarks added

`model/cpu_hotpath_bench_test.go` adds synthetic baselines for:

- `BenchmarkCPUHotRMSNorm3584`
- `BenchmarkCPUHotGELUTanhMul8192`
- `BenchmarkCPUHotSiLUMul8192`
- `BenchmarkCPUHotVecScale3584`
- `BenchmarkCPUHotRoPEPartialGemma4SWA`
- `BenchmarkCPUHotGQAAttentionDecode512`
- `BenchmarkCPUHotGemvMLQ1536x2048`

Run with:

```bash
go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```

## Dispatch cleanup status

- `RuntimeCapabilities()` in the `backends/simd` package centralizes architecture/runtime feature reporting.
- `simd.HasSgemmAsm`, `simd.HasDotAsm`, and `simd.HasVecAsm` expose runtime-safe capability gates.
- `Sdot`/`Saxpy` now dispatch through small Go wrappers and fall back to scalar code if AVX2/FMA or NEON is unavailable, or if callers pass mismatched lengths.
- SGEMM callers continue to check `simd.HasSgemmAsm` before invoking assembly kernels; tensor matmul helpers avoid passing zero-length slice pointers to SIMD entrypoints. `SgemmNTGebp` and `SgemmNTBlockedFMA` now validate dimensions, pointers, strides, and overflow before unsafe slicing/pointer arithmetic.
- Vector entrypoints (`VecAdd`, `VecMul`, `VecScaleAdd`, `RMSNorm*`, `ToBF16`, BF16 helpers) now dispatch through Go wrappers and fall back to scalar code when runtime SIMD gates are false. Scalar fallbacks bound all participating slices and leave untouched destination tails unchanged on malformed inputs.
- Activation wrappers (`VecSiLUMul`, `GELUTanhMul`) intentionally call Go math directly until polynomial SIMD approximations are implemented; prior assembly stubs only bounced back into Go.

## Baseline snapshot (i7-12700, amd64)

```text
BenchmarkCPUHotRMSNorm3584              ~0.50 µs/op, 0 allocs
BenchmarkCPUHotGELUTanhMul8192          ~187 µs/op, 0 allocs
BenchmarkCPUHotSiLUMul8192              ~69 µs/op, 0 allocs
BenchmarkCPUHotVecScale3584             ~0.17 µs/op, 0 allocs
BenchmarkCPUHotRoPEPartialGemma4SWA     ~2.6 µs/op, 0 allocs
BenchmarkCPUHotGQAAttentionDecode512    ~0.25 ms/op, 0 allocs  (caller-owned scratch + Sdot/Saxpy)
BenchmarkCPUHotGemvMLQ1536x2048         ~10.4 ms/op, 0 allocs
```

## Immediate next steps

1. Vectorize `RoPEPartial` on AVX2 and NEON.
2. Add MLX4 GEMV SIMD kernels for CPU quantized decode and MoE experts.
3. Extend caller-owned scratch-buffer reuse to MLP, PLI, MoE, and TurboQuant decode paths.
4. Add allocation gates for decode once broader scratch-buffer reuse lands.
5. Runtime-verify arm64 NEON kernels on Orange Pi 6+.


## Folder reorg note

The current `backends/simd` package keeps architecture-specific files in one package with Go build tags (`*_amd64.go`, `*_arm64.go`, `*_other.go`). Phase 6.6 has started with facade-preserving cleanup: scalar dot/SAXPY fallbacks are now in `scalar.go`, scalar RMSNorm uses precise `math.Sqrt`, BF16 GEMV validates shape-product overflow, and SGEMM/GEBP/gather wrappers validate capability gates and overflow-prone pointer arithmetic. A literal subfolder split would create separate Go packages, so keep `backends/simd` as the public facade and split internals only after wrapper boundaries are explicit. See `docs/simd-folder-reorg.md`.
