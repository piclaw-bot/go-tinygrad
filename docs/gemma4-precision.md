# Gemma4 GPU Precision Characterization

This document describes the known CPU↔GPU numerical precision gap for
**Gemma4-E2B MLX 4-bit** inference and the mitigations explored.

## Summary

Gemma4 is significantly more precision-sensitive than earlier architectures
(Gemma3, Qwen2, LLaMA). With MLX 4-bit quantized weights, the GPU and CPU
paths produce slightly different intermediate results due to GEMV accumulation
order differences. These tiny per-layer differences get amplified through the
35-layer stack, eventually producing different top-1 tokens.

**Gemma3 MLX4 does not have this problem** — it is less precision-sensitive.
All other supported models (SmolLM2, Qwen2.5, Qwen3) also work correctly on GPU.

## Root Cause

GPU GEMV kernels use **parallel shared-memory reduction** (256 threads per block),
while CPU `GemvMLQ` uses **sequential scalar accumulation**. Floating-point
addition is non-associative, so different accumulation orders produce different
results — typically at the BF16 precision boundary (~7 bits of mantissa).

For Gemma4 specifically:
- The model uses **per-layer input gating**, **dual RoPE**, **V norm no-scale**,
  **KV sharing**, and **layer scalars** that create a long chain of precision-
  sensitive operations
- Layer-0 projection/attention differences are small (worst-step maxAbs: `q` 0.25,
  `attn` 0.034, `o` 0.022, `mlp_input` 0.031) but sufficient to seed drift
- By layer 7–8, the cumulative divergence reaches maxAbs > 1.0 and continues growing
- Each local block (post-attention, MLP, PLI, layer-scalar tail) is self-consistent
  when fed the GPU's own intermediate state
- Replaying the CPU run from the GPU's layer-0 output for all prompt steps
  substantially reduces downstream drift and recovers the GPU top-1 token

## Bugs Fixed

Two real CPU/GPU **semantic mismatches** were found and fixed during this investigation:

1. **`PreFFNNorm` missing BF16 truncation** — GPU was feeding higher-precision
   `mlp_input` than the CPU path (which uses `RMSNormBF16`). Fixed in `gpu_forward.go`.

2. **`VNormNoScale` extra BF16 truncation** — GPU was truncating `v` after norm,
   while the CPU path truncates `v` before V-norm but not after. Fixed in `gpu_forward.go`.

Both fixes materially improved the early-layer drift (layer-0 maxAbs improved from
0.125 to 0.0039 after both fixes).

## Diagnostic Evidence

### Layer-by-layer hidden-state drift (quantized CPU vs GPU, after BF16 fixes)

| Layer | maxAbs | meanAbs |
|-------|--------|---------|
| 0     | 0.0039 | 2.47e-5 |
| 1     | 0.0078 | 0.0012  |
| 5     | 0.49   | 0.073   |
| 7     | 1.68   | 0.293   |
| 8     | 5.88   | 0.292   |
| 14    | 8.75   | 1.31    |
| 15    | 15.9   | 1.89    |
| 34    | 3.95   | 0.645   |
| logits| 98.7   | 14.5    |

### Self-consistency audit (layers 0–8)

Every local block was tested by feeding the GPU's own captured intermediate
state to a CPU or fresh-GPU recomputation:

- **Post-attention** (PostNorm → residual → PreFFNNorm → BF16): exact match (maxAbs = 0) at every tested layer
- **MLP** (gate/up from `mlp_input`, GELU, down): close to exact at every layer
- **PLI** (per-layer input gating): exact on fresh GPU, very close on CPU recompute
- **Layer-scalar tail**: exact (`BF16(scalar × hidden_post_pli)` matches final hidden)
- **Layer handoff**: exact (final hidden == next layer's `hidden_in`) at every boundary

### Forward sensitivity replay

| Replay condition | Layer 14 drift | Layer 15 drift | GPU top-1 recovered? |
|---|---|---|---|
| Baseline CPUq | 9.25 | 14.60 | ❌ |
| From GPU layer-0 `hidden_in` | 9.25 | 14.60 | ❌ (no effect) |
| From GPU `pli0_input` | 9.25 | 14.60 | ❌ (no effect) |
| From GPU layer-0 `mlp_input` | 5.96 | 10.19 | partial |
| From GPU layer-0 output | **3.63** | **7.45** | ✅ |

## CPU Output Quality

Both CPU paths (dequantized and quantized) produce garbled text for Gemma4-E2B:

- **Dequantized CPU**: `by其実の事Essaオ Nulla บ้าง epicPlease fullEXTRA **- and **のア쿨的热C-`
- **Quantized CPU**: `ខាង책aCJa bởidist đầyparatssä Astrid一家 more----------------- ไปด้วย **(便 estampado`

This means the GPU precision gap, while real and now well-characterized, is
**not the only correctness blocker**. There is likely a remaining CPU-side
architectural issue (e.g. SWA window masking during generation, KV cache
ordering with shared layers, or a generation-loop bug) that also needs
to be resolved.

Gemma3-1B works correctly on CPU, so the issue is Gemma4-specific.

### 1. Use native MLX GEMV exclusively (1.4a)
Ensure all Gemma4 projections use `GemvMLXDirect` with true native MLX weight
buffers, avoiding any fallback to GPTQ-transposed layout. The native path may
have better accumulation parity with CPU `GemvMLQ`.

### 2. Use BF16 model instead of MLX4 (1.4b)
BF16 weights use `DevGemv` (cuBLAS-style F32 SGEMM), which has much tighter
CPU↔GPU parity than quantized GEMV. Gemma3-1B BF16 already works correctly at
19 tok/s. If a Gemma4 BF16 variant is available, this sidesteps the quantization
sensitivity entirely.

### 3. Early-layer high-precision accumulation (1.4c)
Use Kahan summation or FP64 accumulation in the GEMV kernel for layers 0–2 only.
Or run the first 2–3 layers on CPU (sequential accumulation) and switch to GPU
from layer 3+. Small speed cost, potentially large precision improvement where
it matters most.

## Files

- `model/gpu_forward.go` — GPU forward path with BF16 parity fixes
- `model/llama.go` — CPU forward path with debug hooks
- `model/debug_hooks.go` — diagnostic override hooks
- `model/gemma4_quantized_*_test.go` — ~50 env-gated diagnostic tests
- `gpu/devbuf.go` — DevBuf with `DevToBF16` kernel
