# Inference Cache Modes

`InferenceConfig.cache_mode` selects how the prompt and generated tokens interact with cache state.

## `audit`

`audit` is the conservative reference mode.

Role:
- Builds inspection-friendly cache summaries.
- Uses full model forward passes for reference logits.
- Works for MHA, HCA, CSA, and hybrid attention.

Use when:
- Auditing correctness.
- Comparing full-context logits to cached logits.
- Inspecting generated cache metadata without requiring active decode behavior.

## `mha_decode`

`mha_decode` is the active KV-cache path for baseline multi-head attention.

Role:
- Stores real per-layer MHA keys and values.
- Decodes one token at a time from cached K/V tensors.

Constraints:
- Only valid when every attention layer is baseline MHA.
- HCA/CSA models should use `deepseek_decode` instead.

## `deepseek_decode`

`deepseek_decode` is the active cache path for DeepSeek-style HCA, CSA, and hybrid CSA/HCA models.

Role:
- Stores layer-wise HCA compressed/global state plus local windows.
- Stores layer-wise CSA compressed main/index state plus local windows.
- Preserves MoE, mHC, and MTP behavior around the decode step.

Recommended flags:

```python
InferenceConfig(
    cache_mode="deepseek_decode",
    deepseek_prefill_mode="parallel",
    cache_dtype="fp32",
    return_cache_stats=True,
)
```

## Prefill Modes

### `parallel`

`parallel` is the default DeepSeek decode prefill.

It runs the prompt once through `DeepSeekV4LM.forward(...)`, captures the real normalized attention input before every attention module, projects those states through each HCA/CSA attention module, and builds compressed/local/pending caches from the full prompt.

This is the closest mode to practical inference prefill because the prompt is not replayed token by token.

### `sequential_debug`

`sequential_debug` calls `model.forward_decode(...)` once per prompt token.

Use it when:
- Debugging per-token cache mutation.
- Comparing layer cache state against the active decode path.
- Confirming that future generated tokens can be processed without full forward calls.

It is intentionally slower and should not be the default for normal generation.

## Cache Statistics

Set `return_cache_stats=True` to return fields such as:

- `cache_mode`
- `tokens_seen`
- `sequence_length`
- `layers_by_cache_type`
- `cache_population`
- `deepseek_active_decode`
- `num_hca_compressed_entries`
- `num_csa_compressed_main_entries`
- `num_hca_pending_tokens`
- `num_csa_pending_tokens`
- `local_window_size`
