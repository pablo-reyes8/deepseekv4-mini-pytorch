from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from inference.cache_utils import concat_optional, crop_last, move_optional, tensors_memory_bytes


def default_hca_compressor(c: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    weights = torch.ones_like(c[..., :1])
    if mask is not None:
        weights = mask.to(device=c.device, dtype=c.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    c_mean = (c * weights).sum(dim=1) / denom
    z_mean = (z * weights).sum(dim=1) / denom
    return 0.5 * (c_mean + z_mean)


@dataclass
class HCALayerCache:
    compressed_kv: Optional[torch.Tensor] = None
    compressed_positions: Optional[torch.Tensor] = None
    compressed_valid_mask: Optional[torch.Tensor] = None

    local_c: Optional[torch.Tensor] = None
    local_positions: Optional[torch.Tensor] = None
    local_valid_mask: Optional[torch.Tensor] = None

    pending_c: Optional[torch.Tensor] = None
    pending_z: Optional[torch.Tensor] = None
    pending_positions: Optional[torch.Tensor] = None
    pending_mask: Optional[torch.Tensor] = None

    compression_factor: int = 1
    local_window_size: Optional[int] = None
    tokens_seen: int = 0

    def append_token_state(
        self,
        c_t: torch.Tensor,
        z_t: Optional[torch.Tensor] = None,
        position_t: Optional[torch.Tensor] = None,
        valid_mask_t: Optional[torch.Tensor] = None,
    ) -> "HCALayerCache":
        if c_t.dim() != 3:
            raise ValueError(f"c_t must have shape [B,1,D], got {tuple(c_t.shape)}")
        if z_t is None:
            z_t = c_t
        if valid_mask_t is None:
            valid_mask_t = torch.ones(c_t.shape[:2], device=c_t.device, dtype=torch.bool)

        self.pending_c = concat_optional(self.pending_c, c_t, dim=1)
        self.pending_z = concat_optional(self.pending_z, z_t, dim=1)
        if position_t is not None:
            self.pending_positions = concat_optional(self.pending_positions, position_t, dim=1)
        self.pending_mask = concat_optional(self.pending_mask, valid_mask_t.bool(), dim=1)

        self.local_c = concat_optional(self.local_c, c_t, dim=1)
        if position_t is not None:
            self.local_positions = concat_optional(self.local_positions, position_t, dim=1)
        self.local_valid_mask = concat_optional(self.local_valid_mask, valid_mask_t.bool(), dim=1)
        self.crop_local_window(self.local_window_size)

        self.tokens_seen += int(c_t.shape[1])
        return self

    def flush_ready_blocks(
        self,
        compressor: Optional[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]] = None,
    ) -> "HCALayerCache":
        if self.pending_c is None:
            return self
        compressor = compressor or default_hca_compressor
        m = max(1, int(self.compression_factor))

        while self.pending_c is not None and self.pending_c.shape[1] >= m:
            c_block = self.pending_c[:, :m]
            z_block = self.pending_z[:, :m] if self.pending_z is not None else c_block
            mask_block = self.pending_mask[:, :m] if self.pending_mask is not None else None
            compressed = compressor(c_block, z_block, mask_block).unsqueeze(1)

            self.compressed_kv = concat_optional(self.compressed_kv, compressed, dim=1)
            if self.pending_positions is not None:
                pos = self.pending_positions[:, m - 1 : m]
                self.compressed_positions = concat_optional(self.compressed_positions, pos, dim=1)
            valid = (
                mask_block.any(dim=1, keepdim=True)
                if mask_block is not None
                else torch.ones(compressed.shape[:2], device=compressed.device, dtype=torch.bool)
            )
            self.compressed_valid_mask = concat_optional(self.compressed_valid_mask, valid, dim=1)
            self._drop_pending_prefix(m)
        return self

    def _drop_pending_prefix(self, length: int) -> None:
        self.pending_c = self._drop_or_none(self.pending_c, length)
        self.pending_z = self._drop_or_none(self.pending_z, length)
        self.pending_positions = self._drop_or_none(self.pending_positions, length)
        self.pending_mask = self._drop_or_none(self.pending_mask, length)

    @staticmethod
    def _drop_or_none(tensor: Optional[torch.Tensor], length: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        tensor = tensor[:, length:]
        return tensor if tensor.shape[1] > 0 else None

    def get_global_kv(self) -> Optional[torch.Tensor]:
        return self.compressed_kv

    def get_local_state(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.local_c, self.local_valid_mask

    def crop_local_window(self, window_size: Optional[int]) -> "HCALayerCache":
        self.local_c = crop_last(self.local_c, window_size, dim=1)
        self.local_positions = crop_last(self.local_positions, window_size, dim=1)
        self.local_valid_mask = crop_last(self.local_valid_mask, window_size, dim=1)
        return self

    def build_from_full_sequence(
        self,
        c: torch.Tensor,
        z: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        compressor_fn,
    ) -> "HCALayerCache":
        if c.dim() != 3 or z.shape != c.shape:
            raise ValueError("HCA full-sequence cache expects c/z with shape [B,T,D].")
        B, T, _ = c.shape
        if positions.shape != (B, T):
            raise ValueError(f"positions must have shape {(B, T)}, got {tuple(positions.shape)}")
        if valid_mask is None:
            valid_mask = torch.ones(B, T, device=c.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=c.device, dtype=torch.bool)
            if valid_mask.shape != (B, T):
                raise ValueError(f"valid_mask must have shape {(B, T)}, got {tuple(valid_mask.shape)}")

        self.reset()
        m = max(1, int(self.compression_factor))
        n_complete = T // m
        compressed = []
        compressed_positions = []
        compressed_masks = []

        for idx in range(n_complete):
            start = idx * m
            end = start + m
            block = compressor_fn(
                c[:, start:end],
                z[:, start:end],
                valid_mask[:, start:end],
                positions[:, start:end],
            )
            compressed.append(block)
            compressed_positions.append(positions[:, end - 1])
            compressed_masks.append(valid_mask[:, start:end].any(dim=1))

        if compressed:
            self.compressed_kv = torch.stack(compressed, dim=1)
            self.compressed_positions = torch.stack(compressed_positions, dim=1)
            self.compressed_valid_mask = torch.stack(compressed_masks, dim=1)

        tail_start = n_complete * m
        if tail_start < T:
            self.pending_c = c[:, tail_start:].detach()
            self.pending_z = z[:, tail_start:].detach()
            self.pending_positions = positions[:, tail_start:].detach()
            self.pending_mask = valid_mask[:, tail_start:].detach()

        window = self.local_window_size or T
        self.local_c = c[:, max(0, T - window) :].detach()
        self.local_positions = positions[:, max(0, T - window) :].detach()
        self.local_valid_mask = valid_mask[:, max(0, T - window) :].detach()
        self.tokens_seen = int(T)
        return self

    def reset(self) -> None:
        self.compressed_kv = None
        self.compressed_positions = None
        self.compressed_valid_mask = None
        self.local_c = None
        self.local_positions = None
        self.local_valid_mask = None
        self.pending_c = None
        self.pending_z = None
        self.pending_positions = None
        self.pending_mask = None
        self.tokens_seen = 0

    def num_tokens_seen(self) -> int:
        return int(self.tokens_seen)

    def memory_bytes(self) -> int:
        return tensors_memory_bytes(
            self.compressed_kv,
            self.compressed_positions,
            self.compressed_valid_mask,
            self.local_c,
            self.local_positions,
            self.local_valid_mask,
            self.pending_c,
            self.pending_z,
            self.pending_positions,
            self.pending_mask,
        )

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "HCALayerCache":
        for name in [
            "compressed_kv",
            "local_c",
            "pending_c",
            "pending_z",
        ]:
            setattr(self, name, move_optional(getattr(self, name), device=device, dtype=dtype))
        for name in [
            "compressed_positions",
            "compressed_valid_mask",
            "local_positions",
            "local_valid_mask",
            "pending_positions",
            "pending_mask",
        ]:
            setattr(self, name, move_optional(getattr(self, name), device=device, dtype=None))
        return self
