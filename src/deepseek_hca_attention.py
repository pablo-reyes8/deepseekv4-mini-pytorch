from src.hca_components import *
from src.transformer_modules.rope import *

# ============================================================
# HCA ATTENTION
# ============================================================

class HCAAttention(nn.Module):
    """
    Heavily Compressed Attention.

    Canonical mini version:

    1. Projects token hidden states into:
        - multi-head queries Q
        - shared KV entries C
        - compression logits Z

    2. Compresses every m' token-level KV entries into one heavily compressed
       KV entry.

    3. Performs dense MQA-style attention from every query to:
        - previous completed compressed KV blocks
        - exact local sliding-window KV tokens
        - optional attention sink KV

    4. Uses a grouped output projection, implemented in PyTorch.

    Input:
        x: [B, T, d_model]

    Output:
        out: [B, T, d_model]

    If need_weights=True:
        out, aux

        aux = {
            "sink_attn_weights":       [B, H, T, 1] if enabled
            "global_attn_weights":     [B, H, T, S]
            "local_attn_weights":      [B, H, T, T]
            "compressed_valid_mask":   [B, S]
            "compressed_position_ids": [S] or [B, S]
        }
    """

    def __init__(self, config: HCAConfig):
        super().__init__()

        config.validate()

        self.config = config

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.d_model // config.n_heads
        )

        self.inner_dim = self.n_heads * self.head_dim

        self.compression_factor = config.compression_factor
        self.window_size = config.window_size
        self.max_seq_len = config.max_seq_len
        self.use_rope = config.use_rope
        self.use_attention_sink = config.use_attention_sink
        self.use_grouped_output_projection = config.use_grouped_output_projection

        # Multi-query attention:
        # Q has H heads, but K/V are shared across heads.
        self.q_proj = nn.Linear(
            self.d_model,
            self.inner_dim,
            bias=config.use_bias,
        )

        self.kv_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        self.z_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        self.compressor = HCATokenCompressor(
            compression_factor=config.compression_factor,
            head_dim=self.head_dim,
            init_std=config.init_std,
        )

        # Optional global attention sink.
        # Shape convention:
        #   sink_k: [1, 1, Dh]
        #   sink_v: [1, 1, Dh]
        #
        # It is shared across batch and heads, exactly as the shared KV path.
        if self.use_attention_sink:
            self.sink_k = nn.Parameter(torch.empty(1, 1, self.head_dim))
            self.sink_v = nn.Parameter(torch.empty(1, 1, self.head_dim))
        else:
            self.sink_k = None
            self.sink_v = None

        # More canonical grouped output projection.
        if self.use_grouped_output_projection:
            num_groups = (
                config.output_projection_groups
                if config.output_projection_groups is not None
                else self.n_heads
            )

            self.out_proj = GroupedOutputProjection(
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                num_groups=num_groups,
                bias=config.use_bias,
                init_std=config.init_std,
            )

            self.final_out_proj = None

        else:
            self.out_proj = nn.Linear(
                self.inner_dim,
                self.d_model,
                bias=config.use_bias,
            )

            self.final_out_proj = self.out_proj

        if self.use_rope:
            # Assumes RotaryEmbedding exists in your project and accepts:
            #   q_or_k: [B, T, H, Dh]
            #   position_ids: None, [T], or [B, T]
            #   start_pos: int
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                rotary_dim=config.rotary_dim,
                base=config.rope_theta,
            )
        else:
            self.rope = None

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [self.q_proj, self.kv_proj, self.z_proj]:
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if not self.use_grouped_output_projection:
            nn.init.normal_(
                self.out_proj.weight,
                mean=0.0,
                std=self.config.init_std,
            )

            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

        if self.use_attention_sink:
            nn.init.normal_(self.sink_k, mean=0.0, std=self.config.init_std)
            nn.init.normal_(self.sink_v, mean=0.0, std=self.config.init_std)

    def _shape_q(self, q: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.shape
        return q.view(B, T, self.n_heads, self.head_dim)

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must have shape [B, T], "
                f"got {tuple(attention_mask.shape)}"
            )

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        return attention_mask

    def _build_global_allowed_mask(
        self,
        T: int,
        S: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build conservative causal mask for compressed global HCA attention.

        A compressed block s summarizes tokens:

            [s * m, ..., min((s + 1) * m, T) - 1]

        To avoid future leakage during parallel training, query token t can only
        attend to compressed blocks that are fully completed before the query's
        current compression block.

        Therefore:

            allowed[t, s] = s < floor(t / m)

        The current compression block is handled by the exact local
        sliding-window branch, not by the compressed global branch.

        Shape:
            allowed: [T, S]
        """
        m = self.compression_factor

        q_pos = torch.arange(T, device=device)       # [T]
        q_block_idx = q_pos // m                     # [T]

        block_idx = torch.arange(S, device=device)   # [S]

        allowed = block_idx[None, :] < q_block_idx[:, None]

        return allowed

    def _build_local_allowed_mask(
        self,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build local sliding-window causal mask.

        allowed[t, s] = s <= t and t - s < window_size

        Shape:
            [T, T]
        """
        q_pos = torch.arange(T, device=device)[:, None]
        k_pos = torch.arange(T, device=device)[None, :]

        causal = k_pos <= q_pos
        in_window = (q_pos - k_pos) < self.window_size

        return causal & in_window

    def _safe_concat_softmax(
        self,
        scores: torch.Tensor,
        allowed_mask: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Safe masked softmax over concatenated sink + global + local scores.

        Args:
            scores:
                [B, H, T, N]

            allowed_mask:
                broadcastable to scores.
                True = allowed.
                False = masked.

        Returns:
            weights:
                [B, H, T, N]
                Rows with no valid keys become exactly zero.
        """
        if allowed_mask.dtype != torch.bool:
            allowed_mask = allowed_mask.bool()

        mask_value = torch.finfo(scores.dtype).min

        masked_scores = scores.masked_fill(~allowed_mask, mask_value)

        weights = F.softmax(masked_scores.float(), dim=dim).to(dtype=scores.dtype)

        weights = weights * allowed_mask.to(dtype=weights.dtype)

        denom = weights.sum(dim=dim, keepdim=True)

        weights = torch.where(
            denom > 0,
            weights / denom.clamp_min(torch.finfo(weights.dtype).tiny),
            torch.zeros_like(weights),
        )

        return weights

    def _build_compressed_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        num_blocks: int,
        device: torch.device,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Build position ids for compressed blocks.

        Convention:
            compressed block position = position of the last token in the block.

        Supports:
            position_ids = None   -> [S]
            position_ids: [T]     -> [S]
            position_ids: [B, T]  -> [B, S]
        """
        m = self.compression_factor

        if position_ids is None:
            positions = []

            for i in range(num_blocks):
                start = i * m
                end = min((i + 1) * m, seq_len)
                positions.append(start_pos + end - 1)

            return torch.tensor(
                positions,
                device=device,
                dtype=torch.long,
            )

        if position_ids.device != device:
            position_ids = position_ids.to(device)

        if position_ids.dim() == 1:
            if position_ids.shape[0] != seq_len:
                raise ValueError(
                    f"position_ids with shape [T] must have length T={seq_len}, "
                    f"got {position_ids.shape[0]}"
                )

            positions = []

            for i in range(num_blocks):
                start = i * m
                end = min((i + 1) * m, seq_len)
                positions.append(position_ids[end - 1])

            return torch.stack(positions, dim=0)

        if position_ids.dim() == 2:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"position_ids with shape [B, T] must be {(batch_size, seq_len)}, "
                    f"got {tuple(position_ids.shape)}"
                )

            positions = []

            for i in range(num_blocks):
                start = i * m
                end = min((i + 1) * m, seq_len)
                positions.append(position_ids[:, end - 1])

            return torch.stack(positions, dim=1)

        raise ValueError(
            "position_ids must be None, shape [T], or shape [B, T], "
            f"got {tuple(position_ids.shape)}"
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------
        if x.dim() != 3:
            raise ValueError(
                f"HCAAttention expects x with shape [B, T, d_model], "
                f"got {tuple(x.shape)}"
            )

        B, T, C_model = x.shape

        if C_model != self.d_model:
            raise ValueError(
                f"Expected x.shape[-1] == d_model={self.d_model}, got {C_model}"
            )

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        if attention_mask is not None:
            attention_mask = self._validate_attention_mask(
                attention_mask=attention_mask,
                batch_size=B,
                seq_len=T,
            )

        # ----------------------------------------------------
        # Projections
        # ----------------------------------------------------
        q = self.q_proj(x)      # [B, T, H * Dh]
        C = self.kv_proj(x)    # [B, T, Dh], shared K/V path
        Z = self.z_proj(x)     # [B, T, Dh], compression logits

        q = self._shape_q(q)   # [B, T, H, Dh]

        # ----------------------------------------------------
        # RoPE on queries
        # ----------------------------------------------------
        if self.rope is not None:
            q = self.rope(
                q,
                position_ids=position_ids,
                start_pos=start_pos,
            )

        # ----------------------------------------------------
        # Compress token-level KV
        # ----------------------------------------------------
        compressed_C, compressed_valid_mask, _ = self.compressor(
            C=C,
            Z=Z,
            attention_mask=attention_mask,
            start_pos=start_pos,
        )

        S = compressed_C.shape[1]

        compressed_position_ids = self._build_compressed_position_ids(
            position_ids=position_ids,
            batch_size=B,
            seq_len=T,
            num_blocks=S,
            device=x.device,
            start_pos=start_pos,
        )

        # ----------------------------------------------------
        # RoPE on compressed keys
        # ----------------------------------------------------
        if self.rope is not None:
            K_global = compressed_C[:, :, None, :]  # [B, S, 1, Dh]

            K_global = self.rope(
                K_global,
                position_ids=compressed_position_ids,
                start_pos=0,
            )

            K_global = K_global[:, :, 0, :]         # [B, S, Dh]
        else:
            K_global = compressed_C

        V_global = compressed_C                    # [B, S, Dh]

        # ----------------------------------------------------
        # Local exact KV branch
        # ----------------------------------------------------
        V_local = C                                # [B, T, Dh]

        if self.rope is not None:
            K_local = C[:, :, None, :]             # [B, T, 1, Dh]

            K_local = self.rope(
                K_local,
                position_ids=position_ids,
                start_pos=start_pos,
            )

            K_local = K_local[:, :, 0, :]          # [B, T, Dh]
        else:
            K_local = C

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------
        q = q.transpose(1, 2)                      # [B, H, T, Dh]

        scores_parts = []
        allowed_parts = []

        # -------------------------
        # Optional attention sink
        # -------------------------
        if self.use_attention_sink:
            K_sink = self.sink_k.expand(B, -1, -1)  # [B, 1, Dh]

            scores_sink = torch.einsum(
                "bhtd,bsd->bhts",
                q,
                K_sink,
            ) / math.sqrt(self.head_dim)             # [B, H, T, 1]

            sink_allowed = torch.ones(
                B,
                self.n_heads,
                T,
                1,
                device=x.device,
                dtype=torch.bool,
            )

            scores_parts.append(scores_sink)
            allowed_parts.append(sink_allowed)

        # -------------------------
        # Global compressed scores
        # -------------------------
        scores_global = torch.einsum(
            "bhtd,bsd->bhts",
            q,
            K_global,
        ) / math.sqrt(self.head_dim)                 # [B, H, T, S]

        global_allowed = self._build_global_allowed_mask(
            T=T,
            S=S,
            device=x.device,
        )                                           # [T, S]

        global_allowed = global_allowed[None, None, :, :]  # [1, 1, T, S]

        compressed_allowed = compressed_valid_mask[:, None, None, :]  # [B, 1, 1, S]

        global_allowed = global_allowed & compressed_allowed          # [B, 1, T, S]
        global_allowed = global_allowed.expand(B, self.n_heads, T, S)

        scores_parts.append(scores_global)
        allowed_parts.append(global_allowed)

        # -------------------------
        # Local exact scores
        # -------------------------
        scores_local = torch.einsum(
            "bhtd,bsd->bhts",
            q,
            K_local,
        ) / math.sqrt(self.head_dim)                 # [B, H, T, T]

        local_allowed = self._build_local_allowed_mask(
            T=T,
            device=x.device,
        )                                           # [T, T]

        local_allowed = local_allowed[None, None, :, :]  # [1, 1, T, T]

        if attention_mask is not None:
            local_key_allowed = attention_mask[:, None, None, :].to(
                device=x.device,
                dtype=torch.bool,
            )

            local_allowed = local_allowed & local_key_allowed          # [B, 1, T, T]

        local_allowed = local_allowed.expand(B, self.n_heads, T, T)

        scores_parts.append(scores_local)
        allowed_parts.append(local_allowed)

        # ----------------------------------------------------
        # Concatenate sink + global + local scores
        # ----------------------------------------------------
        scores = torch.cat(scores_parts, dim=-1)          # [B, H, T, N]
        allowed_mask = torch.cat(allowed_parts, dim=-1)   # [B, H, T, N]

        weights = self._safe_concat_softmax(
            scores=scores,
            allowed_mask=allowed_mask,
            dim=-1,
        )

        weights = self.attention_dropout(weights)

        # ----------------------------------------------------
        # Split attention weights
        # ----------------------------------------------------
        offset = 0

        if self.use_attention_sink:
            weights_sink = weights[..., offset:offset + 1]   # [B, H, T, 1]
            offset += 1
        else:
            weights_sink = None

        weights_global = weights[..., offset:offset + S]     # [B, H, T, S]
        offset += S

        weights_local = weights[..., offset:]                # [B, H, T, T]

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------
        context = torch.zeros(
            B,
            self.n_heads,
            T,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        if self.use_attention_sink:
            V_sink = self.sink_v.expand(B, -1, -1)            # [B, 1, Dh]

            context_sink = torch.einsum(
                "bhts,bsd->bhtd",
                weights_sink,
                V_sink,
            )                                                 # [B, H, T, Dh]

            context = context + context_sink

        context_global = torch.einsum(
            "bhts,bsd->bhtd",
            weights_global,
            V_global,
        )                                                     # [B, H, T, Dh]

        context_local = torch.einsum(
            "bhts,bsd->bhtd",
            weights_local,
            V_local,
        )                                                     # [B, H, T, Dh]

        context = context + context_global + context_local

        # ----------------------------------------------------
        # Merge heads + grouped output projection
        # ----------------------------------------------------
        context = context.transpose(1, 2).contiguous()         # [B, T, H, Dh]

        if self.use_grouped_output_projection:
            out = self.out_proj(context)                      # [B, T, D]
        else:
            context = context.view(B, T, self.inner_dim)       # [B, T, D]
            out = self.out_proj(context)                      # [B, T, D]

        out = self.residual_dropout(out)

        if need_weights:
            aux = {
                "global_attn_weights": weights_global,
                "local_attn_weights": weights_local,
                "compressed_valid_mask": compressed_valid_mask,
                "compressed_position_ids": compressed_position_ids,
            }

            if self.use_attention_sink:
                aux["sink_attn_weights"] = weights_sink

            return out, aux

        return out

    def forward_decode(
        self,
        x_t: torch.Tensor,
        cache,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        if x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(f"HCA forward_decode expects x_t [B,1,D], got {tuple(x_t.shape)}")

        B, T, C_model = x_t.shape
        if C_model != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {C_model}")

        q = self._shape_q(self.q_proj(x_t))
        C = self.kv_proj(x_t)
        Z = self.z_proj(x_t)

        if position_ids is None:
            position_ids = torch.full(
                (B, 1),
                int(cache.tokens_seen),
                device=x_t.device,
                dtype=torch.long,
            )
        else:
            position_ids = position_ids.to(device=x_t.device, dtype=torch.long)

        if self.rope is not None:
            q = self.rope(q, position_ids=position_ids, start_pos=0)

        valid_t = (
            attention_mask[:, -1:].to(device=x_t.device).bool()
            if attention_mask is not None
            else torch.ones(B, 1, device=x_t.device, dtype=torch.bool)
        )

        def real_compressor(c_block, z_block, mask_block=None):
            compressed, _, _ = self.compressor(
                C=c_block,
                Z=z_block,
                attention_mask=mask_block,
                start_pos=0,
            )
            return compressed[:, 0, :]

        cache.append_token_state(C, Z, position_ids, valid_t)
        cache.flush_ready_blocks(compressor=real_compressor)

        q = q.transpose(1, 2)  # [B,H,1,Dh]
        scores_parts = []
        allowed_parts = []
        value_parts = []
        part_names = []

        if self.use_attention_sink:
            K_sink = self.sink_k.expand(B, -1, -1)
            scores_sink = torch.einsum("bhtd,bsd->bhts", q, K_sink) / math.sqrt(self.head_dim)
            scores_parts.append(scores_sink)
            allowed_parts.append(torch.ones(B, self.n_heads, 1, 1, device=x_t.device, dtype=torch.bool))
            value_parts.append(self.sink_v.expand(B, -1, -1))
            part_names.append("sink")

        if cache.compressed_kv is not None and cache.compressed_kv.shape[1] > 0:
            K_global = cache.compressed_kv
            if self.rope is not None:
                K_global_rope = K_global[:, :, None, :]
                K_global_rope = self.rope(
                    K_global_rope,
                    position_ids=cache.compressed_positions,
                    start_pos=0,
                )
                K_global = K_global_rope[:, :, 0, :]

            scores_global = torch.einsum("bhtd,bsd->bhts", q, K_global) / math.sqrt(self.head_dim)
            allowed_global = cache.compressed_valid_mask[:, None, None, :].expand(
                B, self.n_heads, 1, cache.compressed_kv.shape[1]
            )
            scores_parts.append(scores_global)
            allowed_parts.append(allowed_global)
            value_parts.append(cache.compressed_kv)
            part_names.append("global")

        if cache.local_c is not None and cache.local_c.shape[1] > 0:
            K_local = cache.local_c
            if self.rope is not None:
                K_local_rope = K_local[:, :, None, :]
                K_local_rope = self.rope(
                    K_local_rope,
                    position_ids=cache.local_positions,
                    start_pos=0,
                )
                K_local = K_local_rope[:, :, 0, :]

            scores_local = torch.einsum("bhtd,bsd->bhts", q, K_local) / math.sqrt(self.head_dim)
            allowed_local = cache.local_valid_mask[:, None, None, :].expand(
                B, self.n_heads, 1, cache.local_c.shape[1]
            )
            scores_parts.append(scores_local)
            allowed_parts.append(allowed_local)
            value_parts.append(cache.local_c)
            part_names.append("local")

        scores = torch.cat(scores_parts, dim=-1)
        allowed = torch.cat(allowed_parts, dim=-1)
        weights = self._safe_concat_softmax(scores=scores, allowed_mask=allowed, dim=-1)
        weights = self.attention_dropout(weights)

        context = torch.zeros(B, self.n_heads, 1, self.head_dim, device=x_t.device, dtype=x_t.dtype)
        offset = 0
        aux = {}
        for name, values in zip(part_names, value_parts):
            width = values.shape[1]
            part_weights = weights[..., offset : offset + width]
            context = context + torch.einsum("bhts,bsd->bhtd", part_weights, values)
            if need_weights:
                aux[f"{name}_attn_weights"] = part_weights
            offset += width

        context = context.transpose(1, 2).contiguous()
        if self.use_grouped_output_projection:
            out = self.out_proj(context)
        else:
            out = self.out_proj(context.view(B, T, self.inner_dim))
        out = self.residual_dropout(out)

        if need_weights:
            aux["compressed_valid_mask"] = cache.compressed_valid_mask
            aux["compressed_position_ids"] = cache.compressed_positions
        return out, cache, aux
