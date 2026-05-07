from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from inference.cache_utils import concat_optional, crop_last, move_optional, tensors_memory_bytes


def default_csa_compressor(
    current_a: torch.Tensor,
    previous_b: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    **_: object,
) -> torch.Tensor:
    weights = torch.ones_like(current_a[..., :1])
    if mask is not None:
        weights = mask.to(device=current_a.device, dtype=current_a.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    current = (current_a * weights).sum(dim=1) / denom

    if previous_b is None:
        return current
    previous = previous_b.mean(dim=1)
    return 0.5 * (current + previous)


@dataclass
class CSALayerCache:
    compressed_main: Optional[torch.Tensor] = None
    compressed_index: Optional[torch.Tensor] = None
    compressed_positions: Optional[torch.Tensor] = None
    compressed_valid_mask: Optional[torch.Tensor] = None

    local_c: Optional[torch.Tensor] = None
    local_positions: Optional[torch.Tensor] = None
    local_valid_mask: Optional[torch.Tensor] = None

    pending_a_c: Optional[torch.Tensor] = None
    pending_b_c: Optional[torch.Tensor] = None
    pending_a_z: Optional[torch.Tensor] = None
    pending_b_z: Optional[torch.Tensor] = None

    pending_index_a_c: Optional[torch.Tensor] = None
    pending_index_b_c: Optional[torch.Tensor] = None
    pending_index_a_z: Optional[torch.Tensor] = None
    pending_index_b_z: Optional[torch.Tensor] = None

    pending_positions: Optional[torch.Tensor] = None
    pending_mask: Optional[torch.Tensor] = None

    previous_b_c: Optional[torch.Tensor] = None
    previous_b_z: Optional[torch.Tensor] = None
    previous_index_b_c: Optional[torch.Tensor] = None
    previous_index_b_z: Optional[torch.Tensor] = None
    previous_mask: Optional[torch.Tensor] = None

    compression_factor: int = 1
    local_window_size: Optional[int] = None
    tokens_seen: int = 0

    def append_token_state(
        self,
        a_c_t: torch.Tensor,
        b_c_t: Optional[torch.Tensor] = None,
        a_z_t: Optional[torch.Tensor] = None,
        b_z_t: Optional[torch.Tensor] = None,
        index_a_c_t: Optional[torch.Tensor] = None,
        index_b_c_t: Optional[torch.Tensor] = None,
        index_a_z_t: Optional[torch.Tensor] = None,
        index_b_z_t: Optional[torch.Tensor] = None,
        position_t: Optional[torch.Tensor] = None,
        valid_mask_t: Optional[torch.Tensor] = None,
    ) -> "CSALayerCache":
        if a_c_t.dim() != 3:
            raise ValueError(f"a_c_t must have shape [B,1,D], got {tuple(a_c_t.shape)}")

        b_c_t = b_c_t if b_c_t is not None else a_c_t
        a_z_t = a_z_t if a_z_t is not None else a_c_t
        b_z_t = b_z_t if b_z_t is not None else b_c_t
        index_a_c_t = index_a_c_t if index_a_c_t is not None else a_c_t
        index_b_c_t = index_b_c_t if index_b_c_t is not None else index_a_c_t
        index_a_z_t = index_a_z_t if index_a_z_t is not None else index_a_c_t
        index_b_z_t = index_b_z_t if index_b_z_t is not None else index_b_c_t
        if valid_mask_t is None:
            valid_mask_t = torch.ones(a_c_t.shape[:2], device=a_c_t.device, dtype=torch.bool)

        self.pending_a_c = concat_optional(self.pending_a_c, a_c_t, dim=1)
        self.pending_b_c = concat_optional(self.pending_b_c, b_c_t, dim=1)
        self.pending_a_z = concat_optional(self.pending_a_z, a_z_t, dim=1)
        self.pending_b_z = concat_optional(self.pending_b_z, b_z_t, dim=1)
        self.pending_index_a_c = concat_optional(self.pending_index_a_c, index_a_c_t, dim=1)
        self.pending_index_b_c = concat_optional(self.pending_index_b_c, index_b_c_t, dim=1)
        self.pending_index_a_z = concat_optional(self.pending_index_a_z, index_a_z_t, dim=1)
        self.pending_index_b_z = concat_optional(self.pending_index_b_z, index_b_z_t, dim=1)
        if position_t is not None:
            self.pending_positions = concat_optional(self.pending_positions, position_t, dim=1)
        self.pending_mask = concat_optional(self.pending_mask, valid_mask_t.bool(), dim=1)

        self.local_c = concat_optional(self.local_c, a_c_t, dim=1)
        if position_t is not None:
            self.local_positions = concat_optional(self.local_positions, position_t, dim=1)
        self.local_valid_mask = concat_optional(self.local_valid_mask, valid_mask_t.bool(), dim=1)
        self.crop_local_window(self.local_window_size)

        self.tokens_seen += int(a_c_t.shape[1])
        return self

    def flush_ready_blocks(
        self,
        main_compressor: Optional[
            Callable[..., torch.Tensor]
        ] = None,
        index_compressor: Optional[
            Callable[..., torch.Tensor]
        ] = None,
    ) -> "CSALayerCache":
        if self.pending_a_c is None:
            return self
        main_compressor = main_compressor or default_csa_compressor
        index_compressor = index_compressor or default_csa_compressor
        m = max(1, int(self.compression_factor))

        while self.pending_a_c is not None and self.pending_a_c.shape[1] >= m:
            mask_block = self.pending_mask[:, :m] if self.pending_mask is not None else None
            main = main_compressor(
                self.pending_a_c[:, :m],
                self.previous_b_c,
                mask_block,
                current_z=self.pending_a_z[:, :m] if self.pending_a_z is not None else None,
                previous_z=self.previous_b_z,
                previous_mask=self.previous_mask,
            ).unsqueeze(1)
            index = index_compressor(
                self.pending_index_a_c[:, :m],
                self.previous_index_b_c,
                mask_block,
                current_z=(
                    self.pending_index_a_z[:, :m] if self.pending_index_a_z is not None else None
                ),
                previous_z=self.previous_index_b_z,
                previous_mask=self.previous_mask,
            ).unsqueeze(1)

            self.compressed_main = concat_optional(self.compressed_main, main, dim=1)
            self.compressed_index = concat_optional(self.compressed_index, index, dim=1)
            if self.pending_positions is not None:
                pos = self.pending_positions[:, m - 1 : m]
                self.compressed_positions = concat_optional(self.compressed_positions, pos, dim=1)

            valid = (
                mask_block.any(dim=1, keepdim=True)
                if mask_block is not None
                else torch.ones(main.shape[:2], device=main.device, dtype=torch.bool)
            )
            self.compressed_valid_mask = concat_optional(self.compressed_valid_mask, valid, dim=1)

            self.previous_b_c = self.pending_b_c[:, :m].detach()
            self.previous_b_z = self.pending_b_z[:, :m].detach()
            self.previous_index_b_c = self.pending_index_b_c[:, :m].detach()
            self.previous_index_b_z = self.pending_index_b_z[:, :m].detach()
            self.previous_mask = mask_block.detach() if mask_block is not None else None
            self._drop_pending_prefix(m)
        return self

    def _drop_pending_prefix(self, length: int) -> None:
        for name in [
            "pending_a_c",
            "pending_b_c",
            "pending_a_z",
            "pending_b_z",
            "pending_index_a_c",
            "pending_index_b_c",
            "pending_index_a_z",
            "pending_index_b_z",
            "pending_positions",
            "pending_mask",
        ]:
            tensor = getattr(self, name)
            if tensor is None:
                continue
            tensor = tensor[:, length:]
            setattr(self, name, tensor if tensor.shape[1] > 0 else None)

    def crop_local_window(self, window_size: Optional[int]) -> "CSALayerCache":
        self.local_c = crop_last(self.local_c, window_size, dim=1)
        self.local_positions = crop_last(self.local_positions, window_size, dim=1)
        self.local_valid_mask = crop_last(self.local_valid_mask, window_size, dim=1)
        return self

    def get_global_state(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.compressed_main, self.compressed_index

    def get_local_state(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.local_c, self.local_valid_mask

    def reset(self) -> None:
        for name in [
            "compressed_main",
            "compressed_index",
            "compressed_positions",
            "compressed_valid_mask",
            "local_c",
            "local_positions",
            "local_valid_mask",
            "pending_a_c",
            "pending_b_c",
            "pending_a_z",
            "pending_b_z",
            "pending_index_a_c",
            "pending_index_b_c",
            "pending_index_a_z",
            "pending_index_b_z",
            "pending_positions",
            "pending_mask",
            "previous_b_c",
            "previous_b_z",
            "previous_index_b_c",
            "previous_index_b_z",
            "previous_mask",
        ]:
            setattr(self, name, None)
        self.tokens_seen = 0

    def num_tokens_seen(self) -> int:
        return int(self.tokens_seen)

    def memory_bytes(self) -> int:
        return tensors_memory_bytes(
            self.compressed_main,
            self.compressed_index,
            self.compressed_positions,
            self.compressed_valid_mask,
            self.local_c,
            self.local_positions,
            self.local_valid_mask,
            self.pending_a_c,
            self.pending_b_c,
            self.pending_a_z,
            self.pending_b_z,
            self.pending_index_a_c,
            self.pending_index_b_c,
            self.pending_index_a_z,
            self.pending_index_b_z,
            self.pending_positions,
            self.pending_mask,
            self.previous_b_c,
            self.previous_b_z,
            self.previous_index_b_c,
            self.previous_index_b_z,
            self.previous_mask,
        )

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "CSALayerCache":
        floating = [
            "compressed_main",
            "compressed_index",
            "local_c",
            "pending_a_c",
            "pending_b_c",
            "pending_a_z",
            "pending_b_z",
            "pending_index_a_c",
            "pending_index_b_c",
            "pending_index_a_z",
            "pending_index_b_z",
            "previous_b_c",
            "previous_b_z",
            "previous_index_b_c",
            "previous_index_b_z",
        ]
        for name in floating:
            setattr(self, name, move_optional(getattr(self, name), device=device, dtype=dtype))
        for name in [
            "compressed_positions",
            "compressed_valid_mask",
            "local_positions",
            "local_valid_mask",
            "pending_positions",
            "pending_mask",
            "previous_mask",
        ]:
            setattr(self, name, move_optional(getattr(self, name), device=device, dtype=None))
        return self
