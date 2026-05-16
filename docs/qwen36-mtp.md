# Qwen3.6 27B Native MTP notes

## Goal

Get a Qwen3.6 27B checkpoint with native MTP running in go-pherence as quickly as possible, without confusing it with the separate Gemma4 assistant-drafter MTP or the Orthrus-inspired stock-weight speculative scaffold.

## Public checkpoint finding

Hugging Face search shows active Qwen3.6 27B MTP artifacts, including:

- `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
- `unsloth/Qwen3.6-27B-MTP-GGUF`
- `havenoammo/Qwen3.6-27B-MTP-UD-GGUF`
- `froggeric/Qwen3.6-27B-MTP-GGUF`

The most relevant safetensors checkpoint inspected so far is `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`.

Top-level config:

```text
architectures: [Qwen3_5ForConditionalGeneration]
model_type: qwen3_5
language_model_only: true
text_config.model_type: qwen3_5_text
```

Text config essentials:

```text
hidden_size: 5120
num_hidden_layers: 64
num_attention_heads: 24
num_key_value_heads: 4
head_dim: 256
vocab_size: 248320
layer_types: linear_attention x3, full_attention x1, repeated
mtp_num_hidden_layers: 1
mtp_use_dedicated_embeddings: false
```

Native MTP tensor prefix:

```text
mtp.fc.weight                              BF16 [5120, 10240]
mtp.pre_fc_norm_embedding.weight           BF16 [5120]
mtp.pre_fc_norm_hidden.weight              BF16 [5120]
mtp.layers.0.input_layernorm.weight        BF16 [5120]
mtp.layers.0.self_attn.q_proj.weight       BF16 [12288, 5120]
mtp.layers.0.self_attn.k_proj.weight       BF16 [1024, 5120]
mtp.layers.0.self_attn.v_proj.weight       BF16 [1024, 5120]
mtp.layers.0.self_attn.o_proj.weight       BF16 [5120, 6144]
mtp.layers.0.self_attn.q_norm.weight       BF16 [256]
mtp.layers.0.self_attn.k_norm.weight       BF16 [256]
mtp.layers.0.post_attention_layernorm.weight BF16 [5120]
mtp.layers.0.mlp.gate_proj.weight          BF16 [17408, 5120]
mtp.layers.0.mlp.up_proj.weight            BF16 [17408, 5120]
mtp.layers.0.mlp.down_proj.weight          BF16 [5120, 17408]
mtp.norm.weight                            BF16 [5120]
```

This is not the Gemma4 LiteRT q-only assistant layout. It is an in-model one-layer native MTP head with its own full Q/K/V attention and MLP, plus a fusion/projection matrix over embedding + hidden (`mtp.fc.weight`).

## Important blocker

The inspected safetensors checkpoint is NVFP4:

```text
quantization_config: compressed-tensors/modelopt-style FP4/NVFP4 groups
```

Current go-pherence intentionally rejects public NVFP4 loading/generation until real CPU/CUDA logits/tokens agree. So the fastest route is either:

1. find a BF16/MLX/GPTQ Qwen3.6 27B MTP safetensors checkpoint, or
2. finish enough real-checkpoint NVFP4 loading to run this model, or
3. use GGUF only as a metadata/reference source, not as a direct loader path.

## Shortest implementation path

### Phase A — metadata and loader recognition

- [x] Add `LlamaConfig` fields for Qwen3.5/Qwen3.6 native MTP metadata:
  - `mtp_num_hidden_layers`
  - `mtp_use_dedicated_embeddings`
  - linear/full attention config fields used by Qwen3.5 text models.
- [ ] Add safetensors/header metadata tests for native `mtp.*` tensor detection without downloading weights.
- [ ] Add early clear diagnostic: `qwen3_5_text native MTP detected, base architecture unsupported` instead of failing later on tensor names.

### Phase B — base Qwen3.5/Qwen3.6 model support

Qwen3.6 27B is not plain Qwen3 dense. It uses mixed `linear_attention` / `full_attention` layers and likely `model.language_model.*` nesting.

Needed before MTP can matter:

- load nested `text_config` as `qwen3_5_text`;
- map tensor prefix (`model.language_model.` if present);
- implement or explicitly reject linear-attention layers;
- support attention output gates if required (`attn_output_gate`, `output_gate_type`);
- validate BF16/full-precision baseline first, if a non-NVFP4 checkpoint exists.

### Phase C — native MTP head

Once base forward works:

- define `QwenNativeMTP` with `fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `layers`, and `norm`;
- implement one MTP draft step:
  - normalize embedding and previous hidden separately;
  - concatenate `[embedding || hidden]` to 10240;
  - project through `mtp.fc.weight` to 5120;
  - run the one MTP decoder layer;
  - final `mtp.norm`;
  - use main LM head for logits;
- reuse existing `AcceptMTPDraft` / staged KV commit helpers for verification.

### Phase D — correctness harness

Use `cmd/speccheck` to compare:

- normal greedy generation;
- native-MTP speculative generation with K=1 first;
- then multi-token draft loops.

Do not optimize GPU until CPU token parity is stable.

## Immediate recommendation

For ASAP progress:

1. first find or produce a non-NVFP4 Qwen3.6 27B MTP safetensors artifact;
2. in parallel, add metadata/header tests and explicit unsupported diagnostics for `qwen3_5_text + mtp_num_hidden_layers`;
3. only then implement Qwen3.5 linear-attention base layers;
4. then wire the native MTP head into the existing speculative acceptance/checkpoint harness.
