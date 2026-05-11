# Gemma4 GPU Precision & Correctness

This document describes the Gemma4-E2B MLX 4-bit correctness journey
and the known CPU↔GPU numerical precision characteristics.

## Status: ✅ Working

Gemma4-E2B MLX 4-bit produces correct output on both CPU and GPU:

- **Dequantized CPU**: `Hello! How can I help you today?`
- **Quantized CPU (MLX4)**: `Hello! How can I help you today?`
- **GPU (RTX 3060)**: `Hello! How can I help you today?` — 13.5 tok/s

## Bugs Found and Fixed

### 1. `(1+w)` norm weight offset (root cause of garbled output)

Gemma4 inherits `RMSNorm` from **Gemma3n** (which uses raw `weight`),
not from **Gemma3** (which uses `(1+weight)`). Our loader was incorrectly
adding `+1` to all Gemma4 norm weights (InputNorm, PostNorm, PreFFNNorm,
PostFFNNorm, QNorm, KNorm, final Norm, PerLayerProjNorm, PLIPostNorm).

**Impact**: Complete model failure — garbled multilingual noise on both CPU paths.

**Fix**: Only apply `(1+w)` for `gemma3_text`, not `gemma4_text`.

### 2. `PreFFNNorm` missing BF16 truncation on GPU

GPU was feeding higher-precision `mlp_input` than the CPU path (which uses
`RMSNormBF16`).

**Impact**: Hidden-state drift between CPU and GPU.

**Fix**: Added `DevToBF16` after `PreFFNNorm` on GPU for Gemma3/4.

### 3. `VNormNoScale` extra BF16 truncation on GPU

GPU was truncating `v` after `VNormNoScale`, while the CPU path truncates `v`
before V-norm but not after.

**Impact**: Attention-side precision drift.

**Fix**: Removed extra `DevToBF16` after `VNormNoScale` on GPU.

### 4. Missing SWA window masking

`gqaAttentionScale` had no sliding-window masking for SWA layers. For sequences
longer than `sliding_window` (512), SWA layers should only attend to the most
recent positions.

**Impact**: Incorrect attention for long sequences (not visible on short prompts).

**Fix**: Added window clipping and KV offset for `sliding_attention` layers.

### 5. Layer scalar on CPU (performance)

Layer scalar was applied via CPU `Data()` + multiply + `MarkDirty()`, causing
35 GPU↔CPU round trips per token.

**Impact**: Performance (not correctness).

**Fix**: Replaced with GPU-side `DevScale()`.

## Remaining Precision Characteristics

After all fixes, a small CPU↔GPU numerical gap remains due to GEMV
accumulation order differences (GPU parallel reduction vs CPU sequential).
This gap does **not** affect output quality — both CPU and GPU produce
the same coherent text.

### Layer-by-layer hidden-state drift (quantized CPU vs GPU)

| Layer | maxAbs | meanAbs |
|-------|--------|---------|
| 0     | 0.0039 | 2.47e-5 |
| 7     | 1.68   | 0.293   |
| 14    | 8.75   | 1.31    |
| 34    | 3.95   | 0.645   |

### Self-consistency audit (layers 0–8)

Every local block was tested by feeding the GPU's own captured intermediate
state to a CPU or fresh-GPU recomputation:

- **Post-attention**: exact match (maxAbs = 0) at every tested layer
- **MLP**: close to exact at every layer
- **PLI**: exact on fresh GPU
- **Layer-scalar tail**: exact
- **Layer handoff**: exact at every boundary

## Gemma4-Specific Architecture Features

| Feature | Description | Status |
|---|---|---|
| Per-layer input gating (PLI) | `embed_tokens_per_layer` + projection + GELU gate + norm | ✅ GPU + CPU |
| Layer scalar | Per-layer output scaling (`DevScale` on GPU) | ✅ GPU + CPU |
| KV sharing | Layers 15–34 share K/V from layers 0–14 | ✅ GPU + CPU |
| Dual head dims | SWA: 256, Full: 512 | ✅ |
| Dual RoPE | SWA: theta=10k full, Full: theta=1M partial (25%) | ✅ |
| V norm (no scale) | `RMSNormNoScale` on V projections | ✅ |
| Double-wide MLP | `intermediate_size × 2` for KV-shared layers | ✅ |
| SWA window masking | Clip attention to last `sliding_window` positions | ✅ |
| Chat template | `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n` | ✅ |

## Files

- `model/gpu_forward.go` — GPU forward path with all fixes
- `model/llama.go` — CPU forward path with SWA masking + correct norms
- `model/debug_hooks.go` — diagnostic override hooks
- `model/gemma4_*_test.go` — diagnostic tests behind `//go:build diagnostic` plus `GEMMA4_TRACE_TEST=1`
- `gpu/devbuf.go` — DevBuf with `DevToBF16`, `DevScale`
