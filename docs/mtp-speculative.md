# MTP (Multi-Token Prediction) Speculative Decoding

## Overview

MTP is a speculative decoding technique where a small **drafter model** predicts
several candidate tokens, and the large **verifier model** validates them in one
batched forward pass. If the drafter's predictions match, multiple tokens are
accepted per step — up to 3× speedup.

## Architecture

### Drafter model (Gemma4-E2B-it-assistant)
Local asset: `models/gemma4-e2b-mtp-drafter`.

- Top-level `model_type: gemma4_assistant`, `architectures: [Gemma4AssistantForCausalLM]`
- Nested text config is `model_type: gemma4_text`
- `hidden_size: 256` (vs 1536 in main model)
- `num_hidden_layers: 4` (vs 35 in main model)
- `intermediate_size: 2048`
- `num_attention_heads: 4`, `num_key_value_heads: 1`
- layer types: 3 sliding layers (`head_dim=256`) + 1 full layer (`global_head_dim=512`)
- `num_kv_shared_layers: 4`; all drafter layers consume external/shared KV rather than owning K/V projections
- Disk: 151 MB BF16 safetensors + 31 MB tokenizer
- VRAM estimate: ~50 MB plus runtime buffers

### Special tensors in local safetensors
- `pre_projection.weight`: `[256, 3072]` — maps `embedding(prev_token)[1536] || activation[1536]` into drafter hidden size 256.
- `post_projection.weight`: `[1536, 256]` — maps drafter hidden/projected state back to the main hidden size for the next drafter step and verifier handoff.
- `masked_embedding.centroids.weight`: `[2048, 256]` — centroid embedding table.
- `masked_embedding.token_ordering`: `[262144]` — vocab → centroid ordering/index data.
- `model.embed_tokens.weight`: `[262144, 256]` — drafter-token embedding table.
- Per-layer tensors include only `q_proj`, `q_norm`, `o_proj`, MLP weights, norms, and `layer_scalar`.
- **No `k_proj`, `v_proj`, `k_norm`, or `v_norm` tensors exist in the drafter**; those must come from shared/base-model KV state.

Loader status:
- `LoadLlama(models/gemma4-e2b-mtp-drafter)` fails at `model.layers.0.self_attn.k_proj.weight` because the generic Gemma4 loader assumes owner K/V projections.
- `LoadGemma4MTPDrafter` now loads the local assistant asset into a dedicated q-only drafter structure, including `pre_projection`, `post_projection`, masked embedding tensors, and all four q-only layers.
- Drafter layers mark `KVSourceLayer=-1` because their K/V source is external; the forward pass must explicitly map them to staged/main-model KV state.
- Remaining gap: implement the drafter forward pass with external/shared KV and main-model verifier integration.

### Data flow

```
Main model decode step N:
  → hidden_state[1536]

Drafter loop (K iterations):
  hidden_draft = pre_projection(hidden_state)     [256]
  for layer in drafter_layers:
    hidden_draft = layer(hidden_draft, KV=main_model_KV)
  hidden_main = post_projection(hidden_draft)     [1536]
  logits = LM_head(hidden_main)                   [vocab]
  draft_token = argmax(logits)
  → candidate_tokens[K]

Verifier (main model batched forward):
  Run main model on all K draft tokens in one prefill-style pass
  Compare main model logits with draft tokens
  Accept matching prefix, reject rest
```

### Performance gain
- Drafter: ~0.5ms per token (tiny model, 4 layers, dim=256)
- Verifier: ~5ms for K tokens batched (same as single-token decode)
- If 3/4 draft tokens accepted: 4 tokens for cost of ~7ms vs ~20ms sequential
- Net: ~2.8× throughput improvement

## Implementation plan

1. **Drafter loader** ✅ — parse `gemma4_assistant` top-level config, nested `text_config`, q-only attention blocks, `pre_projection`, `post_projection`, and masked embedding tensors.
2. **Main-model verifier path** — run a short batched forward over `[input_token] + drafted_tokens`, return per-position logits and hidden activations, and stage candidate KV updates.
3. **pre/post projection** — new tensor fields and GEMV wrappers; `pre_projection` consumes concatenated main embedding + activation.
4. **Draft loop** — run drafter for `G` steps, greedily collect candidate tokens, and carry `projected_activations` between draft steps.
5. **Verify** — compare verifier greedy tokens with drafted tokens in one batched pass.
6. **Accept/reject** — keep matching prefix and emit the verifier bonus token on mismatch/all-accepted completion.
7. **KV cache sync** — commit candidate KV for accepted tokens plus bonus token; discard rejected candidate KV tail.
8. **Adaptive K** — track acceptance rate by task/prompt class and adjust draft length.

## Reference Implementations

### Google LiteRT-LM — Gemma 4 single-position MTP

Sources inspected:
- <https://github.com/google-ai-edge/LiteRT-LM>
- `runtime/executor/llm_litert_mtp_drafter.{h,cc}`
- `runtime/executor/llm_litert_compiled_model_executor.cc`
- `schema/capabilities/speculative_decoding.cc`
- HF model cards: `litert-community/gemma-4-E2B-it-litert-lm`, `litert-community/gemma-4-E4B-it-litert-lm`

Key mechanics:
- MTP support is detected from the `.litertlm` bundle: presence of a TFLite model section with `model_type = "tf_lite_mtp_drafter"` marks the file as speculative-decoding-capable.
- The main model exposes a separate `verify` signature. LiteRT-LM infers the draft count from `verify.input_pos` shape: `num_draft_steps = len(input_pos) - 1`.
- The drafter model is compiled separately from the main prefill/decode model, with a `.mtp_drafter` compilation cache suffix.
- The first MTP call after prefill runs a normal single decode first, then passes that decode output's `activations` tensor into the drafter.
- Later MTP calls do not rerun a normal decode first; the drafter keeps enough internal state to restart from the last verifier activation.
- Drafter input `activations` is the concatenation of:
  - token embedding for the previous/generated token, looked up with `EmbeddingLookupManager.LookupDecode`, and
  - projected activations from either the previous drafter step or the verifier output.
  - LiteRT-LM comments describe this as `[B=1, T=1, D=3072]` for E2B/E4B-scale models: embedding `[1536]` + projected activation `[1536]`.
- Each drafter iteration runs the MTP drafter asynchronously, samples greedily from drafter `logits`, and carries forward `projected_activations` for the next drafter step.
- Verification is one batched main-model pass over `[input_token] + drafted_tokens`:
  - fills `input_pos` for `position..position+G`,
  - fills causal mask for `G+1` steps,
  - looks up normal embeddings and Gemma4 per-layer embeddings for all `G+1` tokens,
  - uses the main model's input KV cache and writes the output KV cache for verified/drafted positions.
- Verifier samples `G+1` token IDs from main logits. Accepted tokens are the matching prefix between verifier IDs and drafted IDs. On first mismatch, LiteRT-LM emits the verifier token as a **bonus token**; if all drafts match, it emits verifier ID at index `G` as the bonus token.
- Reported accounting excludes the bonus token: `num_drafted_tokens += G`, `num_verified_tokens += accepted_prefix_len`, and logs success rate on teardown.

Important difference vs the initial go-pherence plan:
- LiteRT-LM's drafter is **not just a standalone tiny autoregressive model**. It consumes the previous token embedding plus a projected activation vector, and the main model needs a verifier path that returns `activations`. This is closer to a hidden-state-conditioned MTP head than a generic auxiliary LM.
- For go-pherence, implement the verifier hidden/activation output first, then wire drafter state. Treat the drafter KV cache question carefully: LiteRT-LM passes base-model KV buffers into the drafter for drafting, and separately uses base-model `verify` to update output KV cache for accepted/drafted candidates.

Performance notes from public HF cards:
- `gemma-4-E2B-it.litertlm`: 2.59 GB bundle; text decoder weights 0.79 GB; embeddings 1.12 GB mmap'd. S26 Ultra GPU baseline 51.5 tok/s; speculative decoding task results: 66.5–91.7 tok/s. CPU baseline 40.7 tok/s; speculative task results: 36.3–47.5 tok/s.
- `gemma-4-E4B-it.litertlm`: 3.66 GB bundle; text decoder weights 2.24 GB; embeddings 0.67 GB mmap'd. S26 Ultra GPU baseline 21.9 tok/s; speculative task results: 36.7–49.4 tok/s. CPU baseline 17.0 tok/s; speculative task results: 21.1–29.5 tok/s.
- The HF cards note speculative decoding is task-dependent because drafter agreement varies with prompt/task.

Implementation implications for go-pherence:
1. Add a main-model `Verify(tokens []int)` path that runs a short batched forward, returns logits for each position, updates candidate KV, and exposes final hidden/projected activations.
2. Preserve or compute Gemma4 per-layer embeddings for all verifier tokens; LiteRT-LM explicitly feeds `per_layer_embeddings` to verifier.
3. Model the drafter input as `embedding(prev_token) || activation`, not just token IDs.
4. Add acceptance accounting with accepted-prefix length plus bonus token, mirroring LiteRT-LM.
5. Consider a `.litertlm` inspector later: section metadata can reveal whether an artifact includes `tf_lite_mtp_drafter`, but implementing the format is not required for native safetensors MTP.

### llama.cpp PR #22673 — MTP for Qwen3.6
- 75% acceptance rate with 3 draft tokens
- >2× speed-up over baseline
- MTP model loads from the same GGUF (not separate)
- Has its own KV cache and context
- Hidden features propagated via "hook" after each ubatch
- Tested on Qwen3.6 27B and 35B-A3B MoE
- `aggregate_accept_rate: 0.8258` in coding benchmark

### Key design decisions from llama.cpp
1. MTP model is a **separate model** but loaded from the **same file**
2. MTP has its **own KV cache** (not shared with main model)
3. Hidden states are extracted via a hook mechanism after each batch
4. Draft tokens verified in a single batched forward pass

## Models with MTP support

| Model | Drafter | Layers | Hidden | Disk |
|---|---|---|---|---|
| Gemma4-E2B | gemma-4-E2B-it-assistant-bf16 | 4 | 256 | 151 MB |
| Gemma4-E4B | gemma-4-E4B-it-assistant | 4 | 256 | ~200 MB |
| Gemma4-26B MoE | gemma-4-26B-A4B-it-assistant | 4 | 256 | ~200 MB |
| Qwen3.6-35B | built-in (mtp_num_hidden_layers=1) | 1 | shared | in-model |
