# MTP (Multi-Token Prediction) Speculative Decoding

## Overview

MTP is a speculative decoding technique where a small **drafter model** predicts
several candidate tokens, and the large **verifier model** validates them in one
batched forward pass. If the drafter's predictions match, multiple tokens are
accepted per step — up to 3× speedup.

## Architecture

### Drafter model (Gemma4-E2B-it-assistant)
- `hidden_size: 256` (vs 1536 in main model)
- `num_hidden_layers: 4` (vs 35 in main model)
- `intermediate_size: 2048`
- `num_kv_shared_layers: 4` (all layers share KV from main model)
- Disk: ~151 MB (BF16)
- VRAM: ~50 MB

### Special tensors
- `pre_projection.weight`: [256, 3072] — maps main model hidden (1536) + context → drafter dim (256)
- `post_projection.weight`: [1536, 256] — maps drafter hidden (256) → main model dim (1536)
- `masked_embedding.centroids.weight`: [2048, 256] — token embedding clustering for efficiency
- `masked_embedding.token_ordering`: [262144] — maps vocab → centroid indices

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

1. **Load drafter alongside main model** — second `LoadLlama` call
2. **pre/post projection** — new tensor fields in model struct
3. **Draft loop** — run drafter K times, collect candidate tokens
4. **Verify** — batched prefill of K candidates through main model
5. **Accept/reject** — compare logits, keep matching prefix
6. **KV cache sync** — drafter shares main model's KV cache
7. **Adaptive K** — track acceptance rate, adjust draft length

## Reference Implementations

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
