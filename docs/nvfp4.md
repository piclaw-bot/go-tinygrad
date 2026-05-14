# NVFP4 / FP4 Support Track

## Why it matters

NVFP4 is now appearing in public NVIDIA Model Optimizer / TensorRT-oriented
checkpoints for both Gemma and Qwen families. For go-pherence this is relevant
because the current fastest dense path is MLX4 on CUDA, while Qwen3 MoE remains
expert-cache/upload limited. Native NVFP4 could become a better GPU-resident
format for large dense and MoE models once loader/runtime/kernel support exists.

## Current repository status

- No NVFP4 loader/runtime support yet.
- Existing 4-bit support is MLX affine int4 and GPTQ int4 in `runtime/quant`,
  with CUDA upload/dispatch in the transitional `gpu` package.
- Existing CUDA BF16/native helpers target Ampere+ conversion paths, but NVFP4
  tensor-core acceleration is a distinct Blackwell-era format path and should be
  treated as a new quantization format, not as a tweak to MLX/GPTQ.
- Vulkan remains a portability track; initial NVFP4 work should target CUDA only.

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

- Detect NVFP4/FP4 from `quantization_config`, `quantize_config.json`,
  `compressed-tensors` metadata, TensorRT Model Optimizer metadata, and tensor
  naming/layout conventions.
- Add safetensors raw dtype/layout inspection tests using small synthetic files
  before attempting full model loads.
- Define an internal `runtime/quant.NVFP4Weight` representation only after
  inspecting real tensor shapes/metadata from at least one Qwen and one Gemma
  checkpoint.

### Confirmed metadata/layout snapshot

Metadata-only inspection on 2026-05-14 confirmed the common ModelOpt NVFP4 tensor family without downloading full shards:

- Quantized linear tensors use `<prefix>.weight` as safetensors `U8` with shape `[outDim, inDim/2]`.
- `<prefix>.weight_scale` is `F8_E4M3` with shape `[outDim, inDim/16]`, confirming 16-value groups for inspected Qwen3/Gemma4 tensors.
- `<prefix>.weight_scale_2` is scalar `F32`; `<prefix>.input_scale` is scalar `F32` on inspected Qwen3 tensors and many Gemma tensors.
- Embeddings and inspected LM heads remain `BF16` in NVIDIA Qwen3 NVFP4 checkpoints.
- Prefixes differ by family: Qwen uses `model.layers...`; NVIDIA/Red Hat Gemma4 use `model.language_model.layers...`.
- Example observed layouts: Qwen3-8B `q_proj.weight U8 [4096,2048]` + `weight_scale F8_E4M3 [4096,256]`; Qwen3-30B-A3B expert `down_proj.weight U8 [2048,384]` + scale `[2048,48]`; Gemma4 dense `down_proj.weight U8 [5376,10752]` + scale `[5376,1344]`.

### CPU fallback

- Implement a correctness-first CPU unpack/dequant path for NVFP4/FP4 weights.
- Use it only for validation and maybe tiny models; CPU performance is not the
  primary target for NVFP4.
- Add golden tests against known decoded values once the exact scale/block layout
  is confirmed.

### CUDA path

- Add CUDA upload representation separate from MLX/GPTQ.
- Prefer Blackwell-native NVFP4 tensor-core paths when available, but provide a
  dequant-to-F16/BF16/F32 fallback kernel for non-Blackwell NVIDIA GPUs.
- Integrate with dense GEMV/GEMM, LM-head, and MoE expert weights separately.
- For Qwen3 MoE, prioritize expert-cache/prefetch behavior together with NVFP4;
  the current bottleneck is cold-miss/upload, so smaller weights help only if
  upload and cache policy are redesigned.

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
