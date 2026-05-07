from __future__ import annotations

from typing import Optional

import torch

from inference.csa_cache import CSALayerCache
from inference.hca_cache import HCALayerCache
from inference.mha_cache import MHACache


class DeepSeekActiveCacheBuilder:
    def __init__(self, cache, inference_config):
        self.cache = cache
        self.cfg = inference_config

    def capture_layer_input(
        self,
        layer_idx: int,
        attention_type: str,
        attention_module: torch.nn.Module,
        x_norm: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        layer_cache = self.cache.layer_caches[layer_idx]

        if attention_type == "mha":
            if not isinstance(layer_cache, MHACache):
                raise TypeError("MHA attention requires MHACache.")
            if not all(hasattr(attention_module, name) for name in ("k_proj", "v_proj", "_shape_projection")):
                raise NotImplementedError("MHA parallel prefill requires k/v projection helpers.")
            layer_cache.reset()
            k = attention_module._shape_projection(attention_module.k_proj(x_norm))
            v = attention_module._shape_projection(attention_module.v_proj(x_norm))
            if getattr(attention_module, "rope", None) is not None:
                k = attention_module.rope(k, position_ids=position_ids, start_pos=0)
            layer_cache.append(k.transpose(1, 2), v.transpose(1, 2), position_ids)
            return

        if attention_type == "hca":
            if not isinstance(layer_cache, HCALayerCache):
                raise TypeError("HCA attention requires HCALayerCache.")
            if not hasattr(attention_module, "project_cache_states_full"):
                raise NotImplementedError("HCA parallel prefill requires project_cache_states_full.")
            states = attention_module.project_cache_states_full(x_norm)
            layer_cache.build_from_full_sequence(
                c=states["c"],
                z=states["z"],
                positions=position_ids,
                valid_mask=attention_mask,
                compressor_fn=attention_module.compress_hca_block_for_cache,
            )
            return

        if attention_type == "csa":
            if not isinstance(layer_cache, CSALayerCache):
                raise TypeError("CSA attention requires CSALayerCache.")
            if not hasattr(attention_module, "project_cache_states_full"):
                raise NotImplementedError("CSA parallel prefill requires project_cache_states_full.")
            states = attention_module.project_cache_states_full(x_norm)
            layer_cache.build_from_full_sequence(
                a_c=states["a_c"],
                b_c=states["b_c"],
                a_z=states["a_z"],
                b_z=states["b_z"],
                index_a_c=states["index_a_c"],
                index_b_c=states["index_b_c"],
                index_a_z=states["index_a_z"],
                index_b_z=states["index_b_z"],
                positions=position_ids,
                valid_mask=attention_mask,
                main_compressor_fn=attention_module.compress_csa_main_block_for_cache,
                index_compressor_fn=attention_module.compress_csa_index_block_for_cache,
                local_c=states.get("local_c"),
            )
            return

        raise NotImplementedError(f"Unsupported attention type for DeepSeek cache builder: {attention_type!r}")
