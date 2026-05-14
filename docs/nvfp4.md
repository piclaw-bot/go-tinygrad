# NVFP4 / FP4 Support Track

## Why it matters

NVFP4 is now appearing in public NVIDIA Model Optimizer / TensorRT-oriented
checkpoints for both Gemma and Qwen families. For go-pherence this is relevant
because the current fastest dense path is MLX4 on CUDA, while Qwen3 MoE remains
expert-cache/upload limited. Native NVFP4 could become a better GPU-resident
format for large dense and MoE models once loader/runtime/kernel support exists.

## Current repository status

- NVFP4/FP4 is detected early from Hugging Face quantization metadata and still
  rejected for public model loading/generation until CPU and CUDA smoke tests
  agree on real checkpoint outputs.
- `loader/config` owns reusable quantization metadata parsing plus NVFP4 tensor
  role/companion-name helpers for Qwen/Gemma layouts.
- `runtime/quant` has a correctness-first `NVFP4Weight`, FP4 E2M1 and
  F8_E4M3FN decode helpers, F32 dequantization, direct GEMV fallback, and golden
  synthetic logit tests.
- `gpu` has a separate `GPUNVFP4Weight` upload representation, raw-byte packing,
  CUDA dequant-to-F32 fallback kernel wiring, hardware capability gating for a
  future native tensor-core path, and a dense GEMV integration point that
  currently materializes F32 weights per call.
- Existing 4-bit support remains MLX affine int4 and GPTQ int4. NVFP4 is treated
  as a distinct quantization family, not as an MLX/GPTQ tweak.
- Vulkan remains a portability track; initial NVFP4 work targets CUDA only.

## Public weight availability found

Searches found the following Hugging Face NVFP4/FP4 model families relevant to
Gemma/Qwen:

| Family | Example checkpoint | Notes |
|---|---|---|
| Qwen3 dense | `nvidia/Qwen3-8B-NVFP4` | NVIDIA TensorRT Model Optimizer quantized FP4/NVFP4 checkpoint. |
| Qwen3 dense | `NVFP4/Qwen3-32B-FP4` | Community/NVFP4 namespace FP4 checkpoint. |
| Qwen3 MoE | `nvidia/Qwen3-30B-A3B-NVFP4` | Directly relevant to current Qwen3 30B-A3B MoE work. |
| Qwen3 large MoE | `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4` | Shows NVIDIA is publishing larger MoE NVFP4 artifacts. |
| Qwen3.5 MoE | `nvidia/Qwen3.5-397B-A17B-NVFP4` | Very large NVIDIA Model Optimizer checkpoint; likely beyond local hardware but useful for format study. |
| Gemma4 | `nvidia/Gemma-4-31B-IT-NVFP4` | NVIDIA optimized checkpoint. |
| Gemma4 | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | Community/Red Hat AI safetensors/compressed-tensors checkpoint. |
| Gemma4 | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | Community NVFP4 quantization. |

These names should be rechecked before implementation because HF model naming
and availability can change, and some artifacts may use `compressed-tensors`,
TensorRT-LLM metadata, or Model Optimizer layouts rather than plain MLX/GPTQ-like
safetensors.

## Fit-gap for go-pherence

### Loader and metadata

- Done: detect NVFP4/FP4 from `quantization_config`, ModelOpt, and
  compressed-tensors metadata before normal tensor loading.
- Done: metadata-only inspection covered Qwen3 dense, Qwen3 MoE, and Gemma4
  checkpoints without downloading full weight shards.
- Done: `loader/config` classifies dense, MoE expert, and router prefixes and
  returns standard ModelOpt companion tensor names.

### Confirmed metadata/layout snapshot

Metadata-only inspection on 2026-05-14 confirmed the common ModelOpt NVFP4 tensor family without downloading full shards:

- Quantized linear tensors use `<prefix>.weight` as safetensors `U8` with shape `[outDim, inDim/2]`.
- `<prefix>.weight_scale` is `F8_E4M3` with shape `[outDim, inDim/16]`, confirming 16-value groups for inspected Qwen3/Gemma4 tensors.
- `<prefix>.weight_scale_2` is scalar `F32`; `<prefix>.input_scale` is scalar `F32` on inspected NVIDIA Qwen3 and Gemma4 quantized tensors.
- Embeddings and inspected LM heads remain `BF16` in NVIDIA Qwen3 NVFP4 checkpoints.
- Prefixes differ by family: Qwen uses `model.layers...`; NVIDIA/Red Hat Gemma4 use `model.language_model.layers...`.
- Example observed layouts: Qwen3-8B `q_proj.weight U8 [4096,2048]` + `weight_scale F8_E4M3 [4096,256]`; Qwen3-30B-A3B expert `down_proj.weight U8 [2048,384]` + scale `[2048,48]`; Gemma4 dense `down_proj.weight U8 [5376,10752]` + scale `[5376,1344]`.
- Qwen3 dense mapping (`nvidia/Qwen3-8B-NVFP4`): every decoder layer has seven quantized linear prefixes, `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`. Each prefix has `.weight`, `.weight_scale`, `.weight_scale_2`, and `.input_scale`. K/V projections additionally expose `.k_scale` / `.v_scale` for KV-cache scaling. Embeddings and `lm_head.weight` are BF16, not NVFP4, in the inspected NVIDIA checkpoint.
- Qwen3 MoE mapping (`nvidia/Qwen3-30B-A3B-NVFP4`): config reports 48 layers, 128 experts/layer, top-8 routing, `moe_intermediate_size=768`, and sparse step 1. Attention projections use the same Q/K/V/O NVFP4 convention as dense Qwen. Experts use `model.layers.N.mlp.experts.E.{gate,up,down}_proj` with the same four companions (`.weight`, `.weight_scale`, `.weight_scale_2`, `.input_scale`); layer 0 has expert ids 0..127 and all expert projections have companions. The router `model.layers.N.mlp.gate.weight`, embeddings, and `lm_head.weight` are BF16 in the inspected checkpoint.
- Gemma4 NVIDIA mapping (`nvidia/Gemma-4-31B-IT-NVFP4`): top-level config is `model_type=gemma4` with nested `text_config.model_type=gemma4_text` (60 layers, hidden 5376, head dim 256). Tensor names are under `model.language_model.*`. Only MLP projections are NVFP4 in the inspected checkpoint: `model.language_model.layers.N.mlp.{gate,up,down}_proj` with the same four companions. Self-attention Q/K/V/O, embeddings, and absent/untied LM-head tensors are BF16, so dense attention NVFP4 loading must remain optional per prefix rather than assumed from model-level quantization metadata.

### CPU fallback

- Done: correctness-first FP4 unpack/dequant and GEMV fallback exists in
  `runtime/quant`.
- Intended use remains validation and tiny synthetic tests; CPU performance is
  not the primary target for NVFP4.
- Done: golden tests cover FP4 codebook, F8_E4M3FN scales, dequant, GEMV, and
  tiny synthetic logits against an explicit F32 reference.

### CUDA path

- Done: `GPUNVFP4Weight` uploads packed FP4 bytes and F8 scale bytes separately
  from MLX/GPTQ structures.
- Done: CUDA dequant-to-F32 fallback kernel is wired through the mega-module,
  with CPU-reference fallback if launch/sync/download fails.
- Done: native NVFP4 tensor-core use is behind a compute-capability gate
  (`>= 10.x`) and has no public dispatch until implemented and validated.
- Done: dense GEMV has a first integration point via dequantized F32 materialize
  and dot product. This is intentionally correctness-first and allocates
  `OutDim*InDim*4` bytes per call.
- Pending: packed/native GEMV/GEMM, LM-head if a checkpoint quantizes it, and
  Qwen3 MoE expert-cache/prefetch redesign.

### Memory budgets

- Extend weight-budget estimates with NVFP4 bytes-per-parameter and metadata
  overhead from actual loaded tensors.
- Report NVFP4 resident/layer/expert bytes separately from MLX/GPTQ.
- Re-evaluate auto-fit and expert-slot sizing using Qwen3-30B-A3B-NVFP4.

## Roadmap insertion

1. **Recon** — download or inspect metadata for one Qwen3 dense, one Qwen3 MoE,
   and one Gemma4 NVFP4 checkpoint; document tensor names/shapes/metadata.
2. **Detection** — add loader detection and explicit unsupported-format errors so
   NVFP4 checkpoints fail clearly instead of being mistaken for MLX/GPTQ.
3. **CPU decode prototype** — correctness-first unpack/dequant tests.
4. **CUDA prototype** — dequant/GEMV path first; native NVFP4 tensor-core path
   later and hardware-gated.
5. **MoE integration** — combine NVFP4 expert weights with expert cache/prefetch
   redesign for Qwen3 MoE.
6. **Budget integration** — placement/memory reports include NVFP4 tensor class
   and metadata overhead.

## Validation policy

- Start with synthetic unit tests and metadata-only inspection.
- Do not make NVFP4 the default for any model until CPU fallback and CUDA smoke
  tests agree on logits/tokens for a small prompt.
- Gate native NVFP4 CUDA kernels by detected hardware capability; provide clear
  fallback/unsupported messages elsewhere.
