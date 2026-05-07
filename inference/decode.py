from __future__ import annotations

from typing import Any, Optional

import torch

from inference.cache_utils import (
    context_window_for_model,
    default_valid_mask,
    hidden_to_index_state,
    hidden_to_mha_kv,
    normalize_position_ids,
    resolve_device,
    token_hidden_state,
)
from inference.csa_cache import CSALayerCache
from inference.hca_cache import HCALayerCache
from inference.hybrid_cache import DeepSeekV4InferenceCache
from inference.inference_config import InferenceConfig
from inference.mha_cache import MHACache


def configure_cache_metadata(cache: DeepSeekV4InferenceCache, cfg: InferenceConfig) -> None:
    cache.metadata["cache_mode"] = cfg.cache_mode
    cache.metadata["active_decode"] = cfg.cache_mode in {"mha_decode", "deepseek_decode"}
    cache.metadata["logits_from_cache"] = cfg.cache_mode in {"mha_decode", "deepseek_decode"}
    if cfg.cache_mode == "mha_decode":
        cache.metadata["cache_population"] = "active_mha_kv"
    elif cfg.cache_mode == "deepseek_decode":
        cache.metadata["cache_population"] = cfg.deepseek_cache_population
    else:
        cache.metadata["cache_population"] = "embedding_proxy"


def _layer_hidden_for_cache(
    model: torch.nn.Module,
    input_ids_t: torch.Tensor,
    cache: DeepSeekV4InferenceCache,
) -> torch.Tensor:
    return token_hidden_state(
        model,
        input_ids_t,
        device=cache.device,
        dtype=cache.dtype,
    )


def update_cache_with_token(
    model: torch.nn.Module,
    input_ids_t: torch.Tensor,
    cache: DeepSeekV4InferenceCache,
    position_ids_t: Optional[torch.Tensor] = None,
    attention_mask_t: Optional[torch.Tensor] = None,
    inference_config: Optional[InferenceConfig] = None,
) -> DeepSeekV4InferenceCache:
    cfg = inference_config or InferenceConfig()
    cfg.validate()
    configure_cache_metadata(cache, cfg)

    input_ids_t = input_ids_t.to(device=cache.device, dtype=torch.long)
    if input_ids_t.dim() != 2 or input_ids_t.shape[1] != 1:
        raise ValueError(f"input_ids_t must have shape [B,1], got {tuple(input_ids_t.shape)}")

    position_ids_t = normalize_position_ids(
        position_ids_t,
        batch_size=input_ids_t.shape[0],
        seq_len=1,
        start_pos=cache.tokens_seen,
        device=cache.device,
    )
    pad_token_id = cfg.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(model, "pad_token_id", None)

    valid_mask_t = (
        attention_mask_t.to(device=cache.device).bool()
        if attention_mask_t is not None
        else default_valid_mask(input_ids_t, pad_token_id)
    )
    hidden = _layer_hidden_for_cache(model, input_ids_t, cache)
    index_dim = int(getattr(getattr(model, "config", None), "indexer_dim", hidden.shape[-1]))

    for layer_cache in cache.layer_caches:
        if isinstance(layer_cache, MHACache):
            k_t, v_t = hidden_to_mha_kv(model, hidden)
            layer_cache.append(k_t, v_t, position_ids_t)
            layer_cache.crop(cfg.max_cache_length)
        elif isinstance(layer_cache, HCALayerCache):
            layer_cache.append_token_state(hidden, hidden, position_ids_t, valid_mask_t)
            if cfg.compress_on_block_ready:
                layer_cache.flush_ready_blocks()
        elif isinstance(layer_cache, CSALayerCache):
            index_state = hidden_to_index_state(hidden, index_dim)
            layer_cache.append_token_state(
                a_c_t=hidden,
                b_c_t=hidden,
                a_z_t=hidden,
                b_z_t=hidden,
                index_a_c_t=index_state,
                index_b_c_t=index_state,
                index_a_z_t=index_state,
                index_b_z_t=index_state,
                position_t=position_ids_t,
                valid_mask_t=valid_mask_t,
            )
            if cfg.compress_on_block_ready:
                layer_cache.flush_ready_blocks()

    cache.append_input_ids(input_ids_t, valid_mask_t.long())
    cache.crop_sequence(cfg.max_cache_length)
    return cache


def _full_forward_decode(
    model: torch.nn.Module,
    cache: DeepSeekV4InferenceCache,
    *,
    return_aux: bool,
) -> dict[str, Any]:
    if cache.sequence_ids is None:
        raise ValueError("cache.sequence_ids is empty; call prefill or update_cache_with_token first.")

    max_context = context_window_for_model(model, None)
    input_ids = cache.sequence_ids[:, -max_context:]
    attention_mask = cache.attention_mask[:, -input_ids.shape[1] :] if cache.attention_mask is not None else None
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_aux=return_aux,
    )
    return {
        "logits": outputs["logits"][:, -1:, :],
        "hidden_states": outputs.get("hidden_states", None),
        "aux": outputs.get("aux", {}),
    }


@torch.no_grad()
def decode_step(
    model: torch.nn.Module,
    input_ids_t: torch.Tensor,
    cache: DeepSeekV4InferenceCache,
    position_ids_t: Optional[torch.Tensor] = None,
    inference_config: Optional[InferenceConfig] = None,
    return_aux: bool = False,
) -> dict[str, Any]:
    cfg = inference_config or InferenceConfig()
    cfg.validate()

    if cache is None:
        raise ValueError("decode_step requires a DeepSeekV4InferenceCache instance.")

    if cfg.cache_mode in {"mha_decode", "deepseek_decode"}:
        model_cfg = getattr(model, "config", None)
        if cfg.cache_mode == "mha_decode" and getattr(model_cfg, "attention_type", None) != "mha":
            raise NotImplementedError("cache_mode='mha_decode' only supports attention_type='mha'.")
        if not hasattr(model, "forward_decode"):
            raise NotImplementedError("Model does not expose forward_decode for active cached decoding.")

        configure_cache_metadata(cache, cfg)
        decoded = model.forward_decode(
            input_ids_t=input_ids_t,
            cache=cache,
            position_ids_t=position_ids_t,
            attention_mask_t=None,
            return_aux=return_aux,
        )
    else:
        update_cache_with_token(
            model,
            input_ids_t=input_ids_t,
            cache=cache,
            position_ids_t=position_ids_t,
            inference_config=cfg,
        )
        decoded = _full_forward_decode(model, cache, return_aux=return_aux)
        decoded["cache"] = cache
        decoded["aux"] = decoded.get("aux", {})

    if return_aux:
        decoded["aux"]["cache_summary"] = cache.cache_summary()
    return decoded


def next_position_ids(cache: DeepSeekV4InferenceCache) -> torch.Tensor:
    return torch.full(
        (cache.batch_size, 1),
        fill_value=int(cache.tokens_seen),
        device=cache.device,
        dtype=torch.long,
    )


def prepare_input_ids(input_ids: torch.Tensor, model: torch.nn.Module, cfg: InferenceConfig) -> torch.Tensor:
    device = resolve_device(model, cfg.device)
    return input_ids.to(device=device, dtype=torch.long)
