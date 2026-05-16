# Qwen3.6 27B Native MTP roadmap

## Active goal

**Qwen3.6 27B with native MTP is now the active project goal.** Get a Qwen3.6 27B checkpoint with native MTP running in go-pherence as quickly as possible, without confusing it with the separate Gemma4 assistant-drafter MTP or the Orthrus-inspired stock-weight speculative scaffold.

Definition of done for the first useful milestone:

- load Qwen3.6 text config and tensors far enough to reject/route unsupported pieces clearly;
- support the base Qwen3.6 text forward path needed by the 27B checkpoint;
- load the native `mtp.*` head metadata/weights;
- run greedy CPU correctness for at least one short prompt;
- run `speccheck` normal-vs-native-MTP parity with K=1;
- only then optimize CUDA/KV reuse.

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


## llama.cpp `mtp-clean` reference mapping

Reference branch: <https://github.com/am17an/llama.cpp/tree/mtp-clean> at inspected commit `2dff7ff`.

Important implementation points to port/adapt:

### Conversion / tensor naming

`conversion/qwen.py` defines `_Qwen35MtpMixin`:

- extends `block_count` by `mtp_num_hidden_layers`;
- writes `nextn_predict_layers`;
- can split trunk and MTP with `--no-mtp` / `--mtp`;
- remaps HF native MTP tensors:

```text
mtp.layers.0.*                  -> model.layers.{num_hidden_layers}.
mtp.fc.weight                   -> model.layers.{bid}.eh_proj.weight
mtp.pre_fc_norm_embedding.weight -> model.layers.{bid}.enorm.weight
mtp.pre_fc_norm_hidden.weight    -> model.layers.{bid}.hnorm.weight
mtp.norm.weight                  -> model.layers.{bid}.shared_head.norm.weight
```

For go-pherence we can either keep the HF `mtp.*` names directly or mirror this logical mapping internally. Keeping direct HF names is less invasive for safetensors.

### Base Qwen3.5/Qwen3.6 text model

`src/models/qwen35.cpp` is the closest architecture reference. Key points:

- `nextn_predict_layers` are treated as extra decoder blocks appended after the main stack.
- Main forward executes only `n_layer - nextn_predict_layers`. Added metadata helpers to compute main-layer count and classify main full/linear-attention layers vs appended native-MTP layers.
- Recurrent/linear-attention layers are all non-full-attention layers before the MTP tail:

```cpp
n_main = n_layer - nextn_predict_layers
recurrent[i] = i < n_main && ((i + 1) % full_attention_interval != 0)
```

- Full-attention Q projection emits query plus gate: `q_proj` output is `2 * n_heads * head_dim`. Added `Qwen35FullAttentionShapesFor` to validate these dimensions (`[12288,5120]` for Qwen3.6 27B q_proj, gate size 6144).
- Attention output is multiplied by `sigmoid(gate)` before `o_proj`.
- Linear attention is a gated delta net with conv/recurrent state. This is the largest base-model blocker for Qwen3.6. Added `Qwen35LinearAttentionShapesFor` to make the tensor shape contract explicit before implementing forward/state updates.

### Native MTP graph

`graph_mtp` in `src/models/qwen35.cpp` is the core draft-head algorithm for one native MTP block:

1. Inputs are the next-token id and a pre-norm hidden row `h_p`.
2. Token embedding comes from dedicated MTP embeddings if present, otherwise main `tok_embd`.
3. Normalize both streams separately:

```text
h_norm = RMSNorm(h_input, hnorm)
e_norm = RMSNorm(tok_embd, enorm)
concat = [e_norm ; h_norm]
cur = eh_proj(concat)
```

4. Run one full-attention Qwen3.5 decoder block:

```text
attn_norm
q_proj -> split Q and gate
q_norm, k_proj, k_norm, v_proj
MRoPE on Q/K
GQA attention
attn *= sigmoid(gate)
o_proj
residual
post_attention_layernorm
SwiGLU MLP
residual
```

5. Save pre-output-norm hidden for the next MTP draft step.
6. Apply shared head norm (`mtp.norm` or output norm), then main LM head (or dedicated shared head) for logits.

### Runtime MTP loop

`common/speculative.cpp` has `common_speculative_state_draft_mtp`:

- target and draft contexts both expose `embeddings_pre_norm`;
- `process()` mirrors target verification batches into the MTP context and stores pre-norm hidden rows;
- `pending_h[seq]` carries `(h_p, x_{p+1})` across calls;
- `draft()` feeds `(last token, pending_h)` into the MTP context, samples greedy/top-k, then feeds each drafted token back with the previous MTP pre-norm hidden row to draft multiple tokens;
- `accept(seq, n_accepted)` advances `pending_h` to the hidden row corresponding to the accepted verifier position.

For go-pherence, this maps onto:

- exposing the main CPU/GPU decode pre-output-norm hidden (`h_pre_norm`) as part of `DecodeOne`;
- adding a native MTP draft state with `pending_h`;
- using existing `AcceptMTPDraft` semantics to update `pending_h` and KV cache after verification.

## Shortest implementation path

### Metadata inspection helper

Use `cmd/qwenmtpmeta` for local metadata inspection without entering the full model loader:

```bash
go run ./cmd/qwenmtpmeta -model /path/to/qwen3.6-27b-mtp
```

It emits JSON with parsed Qwen native-MTP config metadata, layer counts by type, derived Qwen3.5 full/linear attention shape contracts, and any local `mtp.*` safetensors tensor names when `model.safetensors` is present. If safetensors are available, it also reports missing required native-MTP tensors for the configured MTP layer count.

### Phase A — metadata and loader recognition

- [x] Add `LlamaConfig` fields for Qwen3.5/Qwen3.6 native MTP metadata:
  - `mtp_num_hidden_layers`
  - `mtp_use_dedicated_embeddings`
  - linear/full attention config fields used by Qwen3.5 text models.
- [x] Add config/tensor-name metadata tests for native `mtp.*` tensor detection without downloading weights.
- [x] Add early clear diagnostic driven by reusable Qwen native-MTP metadata: `qwen3_5_text native MTP detected, base architecture unsupported` instead of failing later on tensor names.

### Phase B — base Qwen3.5/Qwen3.6 model support

Qwen3.6 27B is not plain Qwen3 dense. It uses mixed `linear_attention` / `full_attention` layers and likely `model.language_model.*` nesting.

Needed before MTP can matter:

- [x] load nested `text_config` as `qwen3_5_text` for metadata/early diagnostics;
- [x] map tensor prefix (`model.language_model.` if present) and provide a candidate tensor-source wrapper for loader integration;
- [x] add Qwen3.5/Qwen3.6 base layer structs separate from existing Qwen3 layer assumptions;
- [x] implement full-attention Qwen3.5 layer CPU skeleton:
  - q_proj outputs query + gate;
  - split Q and gate;
  - Q/K RMSNorm;
  - RoPE/MRoPE;
  - GQA attention;
  - multiply attention output by `sigmoid(gate)`;
  - o_proj + residual + post-attention RMSNorm + SwiGLU MLP.
- [ ] implement linear-attention/gated-delta-net layer (typed recurrent state and clear unsupported forward stub exist):
  - in_proj_qkvz layout and conversion/reorder (projection shape and split primitive staged);
  - conv1d state (state update and depthwise conv primitives staged);
  - beta/alpha/dt/a recurrent update (alpha/beta projection primitive staged; recurrent math still pending);
  - gated RMSNorm with z;
  - recurrent state cache layout and rollback semantics.
- [ ] support attention output gates if required (`attn_output_gate`, `output_gate_type`);
- [ ] validate BF16/full-precision baseline first, if a non-NVFP4 checkpoint exists.

Linear attention is the critical blocker. The helper `Qwen35LinearAttentionShapesFor` only fixes tensor layout; the actual recurrent delta-net math and state management still need implementation.

### Phase C — native MTP head

Once base forward works:

- [x] define `QwenNativeMTPHead` / layer structs with `fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `layers`, and `norm`, plus synthetic shape validation, a tensor-source loader contract, and a safetensors-backed tensor source for BF16/F32 fixtures;
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

Synthetic command-line harness while real Qwen3.6 loading is gated:

```bash
go run ./cmd/qwenmtpsynth -steps 2
```

This exercises the native-MTP plan/draft/acceptance/stat plumbing over tiny deterministic tensors and exits non-zero on acceptance failure.

## Immediate execution plan

This supersedes the prior Orthrus/stock-weight speculative exploration as the main implementation track.

1. **Checkpoint triage**
   - find or produce a non-NVFP4 Qwen3.6 27B MTP safetensors artifact if possible;
   - keep inspecting NVFP4 metadata so the current public checkpoint remains useful as a shape/layout oracle;
   - do not enable public NVFP4 generation until real logits/tokens agree.
2. **Loader/config gate**
   - add metadata/header tests and explicit unsupported diagnostics for `qwen3_5_text + mtp_num_hidden_layers`;
   - avoid silent partial loads or generic missing-tensor errors.
3. **Base Qwen3.6 text model**
   - implement nested `text_config` loading and tensor prefix mapping;
   - implement or explicitly stage support for `linear_attention` / `full_attention` layers;
   - validate full-attention-only synthetic fixtures before touching large weights.
4. **Native MTP head**
   - load `mtp.fc`, pre-FC norms, one MTP decoder layer, and `mtp.norm`;
   - implement K=1/multi-step greedy draft step (CPU skeleton now covers preprojection plus full-attention/MLP MTP block, RoPE when frequency tables are provided, current K/V return, history-aware attention over past+current MTP KV, final MTP norm + main LM-head logits/argmax, `QwenNativeMTPDraftState`, bounded `DraftSteps`, `QwenNativeMTPPlan`, plan-based adapters to existing `AcceptMTPDraft` from verifier tokens or validated logits, accepted-prefix draft-state commit, and native-MTP stats with aggregation helpers; generation-loop integration remains next);
   - reuse `AcceptMTPDraft` and `speccheck` parity checks.
5. **Correctness harness**
   - extend `speccheck` with a native-MTP mode once the draft step exists (flag placeholder `-qwen-native-mtp` now fails clearly until real `LoadLlama` integration lands; synthetic native-MTP correctness is covered in model tests);
   - store golden token baselines for small prompts;
   - only optimize after CPU parity is stable.
6. **Performance path**
   - replace replay verification with KV-reusing verifier block;
   - then move hot MTP and verifier paths to CUDA.
