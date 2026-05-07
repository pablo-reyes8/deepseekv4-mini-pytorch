# HCA and CSA KV Cache Mechanics

The active DeepSeek cache is not a standard Transformer KV cache. HCA and CSA compress blocks of hidden states, keep a local sliding window, and preserve pending tokens that have not yet filled a compression block.

## Shared Shape

At prefill time, each block captures:

```text
x_norm = block.norm1(hidden_states)
```

That tensor is the same normalized attention input used by the attention module. The cache builder then asks the attention module to project cache states from `x_norm`.

## HCA Cache

HCA stores:

- `compressed_kv`: one compressed vector per completed block.
- `compressed_positions`: position id for each compressed block, using the block's final token.
- `compressed_valid_mask`: whether each compressed block has at least one valid token.
- `pending_c`, `pending_z`: tail states that have not reached `hca_compression_factor`.
- `local_c`: recent token states for the sliding local branch.

Config fields that matter most:

- `hca_compression_factor`: number of tokens summarized into one global compressed entry.
- `window_size`: local sliding-window length.
- `use_attention_sink`: adds a learned sink key/value branch.
- `use_rope`: applies RoPE to query and cache keys by stored positions.

During decode, HCA projects the new token into `C` and `Z`, appends it to the pending/local cache, flushes a compressed block when enough pending tokens exist, and attends over sink, compressed global memory, and local memory.

## CSA Cache

CSA stores:

- `compressed_main`: compressed values used by the selected global branch.
- `compressed_index`: compressed index keys used to choose top-k global blocks.
- `compressed_positions`: position id for each compressed block.
- `compressed_valid_mask`: valid-block mask for sparse selection.
- `previous_b_*`: previous overlapping B states used by the next block compression.
- `pending_a_*`, `pending_b_*`: current tail states waiting for a full block.
- `local_c`: recent local branch states.

Config fields that matter most:

- `compression_factor`: CSA block size.
- `top_k_blocks`: number of compressed blocks selected by the indexer.
- `indexer_dim`: latent dimension of the sparse block indexer.
- `n_indexer_heads`: number of index query heads.
- `query_compression_dim`: query latent size before index scoring.
- `window_size`: local sliding-window length.
- `use_separate_local_kv`: uses a dedicated local KV projection.

During decode, CSA projects the new token into A/B main states and A/B index states. When a block is ready, it compresses the current A block together with the previous B block, mirroring the overlapping compression idea used by the paper-style module.

## Parallel Prefill

`deepseek_prefill_mode="parallel"` builds these caches from the full prompt in one model pass:

```text
full prompt forward
  -> capture x_norm at each layer
  -> HCA project C/Z and compress complete blocks
  -> CSA project A/B main + A/B index states and compress complete blocks
  -> store pending tail and local window
```

This mode does call the model's normal forward once during prefill. After prefill, generated tokens use `forward_decode`.

## Sequential Debug Prefill

`deepseek_prefill_mode="sequential_debug"` fills the same cache objects by replaying the prompt token by token through `forward_decode`.

This is useful because it exercises the same mutation path used by generation, but it is slower and less representative of practical prefill.

## Current Scope

The cache is implemented in pure PyTorch and is CPU-testable. It intentionally does not include custom CUDA kernels, fused attention kernels, paged attention allocators, or production serving schedulers.
