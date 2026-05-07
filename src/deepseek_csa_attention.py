# ============================================================
# CANONICAL CSA ATTENTION
# ============================================================


from src.csa_light_indexer import *
from src.transformer_modules.rope import * 

class CSAAttention(nn.Module):
    """
    Canonical CSA mini implementation.

    Core pieces:
        - Overlapped a/b compression
        - Low-rank shared query path
        - Lightning indexer
        - Top-k causal sparse global attention
        - Local sliding-window branch with separate local KV path
        - Shared KV MQA
        - Optional attention sink
        - Grouped output projection
        - RoPE / partial RoPE

    Input:
        x: [B,T,d_model]

    Output:
        out: [B,T,d_model]

    If need_weights=True:
        out, aux
    """

    def __init__(self, config: CSAConfig):
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
        self.top_k = config.top_k
        self.window_size = config.window_size

        self.indexer_dim = config.indexer_dim
        self.n_indexer_heads = config.n_indexer_heads
        self.query_compression_dim = (
            config.query_compression_dim
            if config.query_compression_dim is not None
            else config.indexer_dim
        )

        self.max_seq_len = config.max_seq_len
        self.use_rope = config.use_rope
        self.use_attention_sink = config.use_attention_sink
        self.use_grouped_output_projection = config.use_grouped_output_projection
        self.use_indexer_score_bias = config.use_indexer_score_bias
        self.use_separate_local_kv = config.use_separate_local_kv

        # ----------------------------------------------------
        # Shared low-rank query path
        # ----------------------------------------------------
        self.q_down_proj = nn.Linear(
            self.d_model,
            self.query_compression_dim,
            bias=config.use_bias,
        )

        self.q_up_proj = nn.Linear(
            self.query_compression_dim,
            self.inner_dim,
            bias=config.use_bias,
        )

        self.index_q_up_proj = nn.Linear(
            self.query_compression_dim,
            self.n_indexer_heads * self.indexer_dim,
            bias=config.use_bias,
        )

        self.index_weight_proj = nn.Linear(
            self.d_model,
            self.n_indexer_heads,
            bias=config.use_bias,
        )

        # ----------------------------------------------------
        # Compressed KV path: a/b branches
        # ----------------------------------------------------
        self.a_kv_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        self.b_kv_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        self.a_z_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        self.b_z_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        # ----------------------------------------------------
        # Separate local exact KV branch
        # ----------------------------------------------------
        if self.use_separate_local_kv:
            self.local_kv_proj = nn.Linear(
                self.d_model,
                self.head_dim,
                bias=config.use_bias,
            )
        else:
            self.local_kv_proj = None

        # ----------------------------------------------------
        # Index key path: a/b branches
        # ----------------------------------------------------
        self.a_index_kv_proj = nn.Linear(
            self.d_model,
            self.indexer_dim,
            bias=config.use_bias,
        )

        self.b_index_kv_proj = nn.Linear(
            self.d_model,
            self.indexer_dim,
            bias=config.use_bias,
        )

        self.a_index_z_proj = nn.Linear(
            self.d_model,
            self.indexer_dim,
            bias=config.use_bias,
        )

        self.b_index_z_proj = nn.Linear(
            self.d_model,
            self.indexer_dim,
            bias=config.use_bias,
        )

        # ----------------------------------------------------
        # Compressors + indexer
        # ----------------------------------------------------
        self.kv_compressor = CSAOverlappedCompressor(
            compression_factor=config.compression_factor,
            dim=self.head_dim,
            init_std=config.init_std,
        )

        self.index_compressor = CSAOverlappedCompressor(
            compression_factor=config.compression_factor,
            dim=self.indexer_dim,
            init_std=config.init_std,
        )

        self.indexer = CSALightningIndexer(
            compression_factor=config.compression_factor,
            top_k=config.top_k,
        )

        # ----------------------------------------------------
        # Optional attention sink
        # ----------------------------------------------------
        if self.use_attention_sink:
            self.sink_k = nn.Parameter(torch.empty(1, 1, self.head_dim))
            self.sink_v = nn.Parameter(torch.empty(1, 1, self.head_dim))
        else:
            self.sink_k = None
            self.sink_v = None

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------
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
        else:
            self.out_proj = nn.Linear(
                self.inner_dim,
                self.d_model,
                bias=config.use_bias,
            )

        if self.use_rope:
            # Assumes RotaryEmbedding exists in your project and accepts:
            #   x: [B,T,H,Dh]
            #   position_ids: None, [T], or [B,T]
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
        modules = [
            self.q_down_proj,
            self.q_up_proj,
            self.index_q_up_proj,
            self.index_weight_proj,
            self.a_kv_proj,
            self.b_kv_proj,
            self.a_z_proj,
            self.b_z_proj,
            self.a_index_kv_proj,
            self.b_index_kv_proj,
            self.a_index_z_proj,
            self.b_index_z_proj,
        ]

        if self.local_kv_proj is not None:
            modules.append(self.local_kv_proj)

        if not self.use_grouped_output_projection:
            modules.append(self.out_proj)

        for module in modules:
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if self.use_attention_sink:
            nn.init.normal_(self.sink_k, mean=0.0, std=self.config.init_std)
            nn.init.normal_(self.sink_v, mean=0.0, std=self.config.init_std)

    def project_cache_states_full(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"project_cache_states_full expects x [B,T,D], got {tuple(x.shape)}")
        return {
            "a_c": self.a_kv_proj(x),
            "b_c": self.b_kv_proj(x),
            "a_z": self.a_z_proj(x),
            "b_z": self.b_z_proj(x),
            "index_a_c": self.a_index_kv_proj(x),
            "index_b_c": self.b_index_kv_proj(x),
            "index_a_z": self.a_index_z_proj(x),
            "index_b_z": self.b_index_z_proj(x),
            "local_c": self.local_kv_proj(x) if self.local_kv_proj is not None else self.a_kv_proj(x),
        }

    def _compress_csa_block_for_cache(
        self,
        compressor,
        current_a: torch.Tensor,
        previous_b: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        *,
        current_z: Optional[torch.Tensor] = None,
        previous_z: Optional[torch.Tensor] = None,
        previous_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_z = current_z if current_z is not None else current_a
        B = current_a.shape[0]

        if previous_b is None:
            tokens = current_a
            scores = current_z + compressor.bias_a[: current_a.shape[1], :].to(
                device=current_a.device,
                dtype=current_z.dtype,
            )[None, :, :]
            valid = mask
        else:
            previous_z = previous_z if previous_z is not None else previous_b
            a_len = current_a.shape[1]
            b_len = previous_b.shape[1]
            a_scores = current_z + compressor.bias_a[:a_len, :].to(
                device=current_a.device,
                dtype=current_z.dtype,
            )[None, :, :]
            b_scores = previous_z + compressor.bias_b[:b_len, :].to(
                device=previous_b.device,
                dtype=previous_z.dtype,
            )[None, :, :]
            tokens = torch.cat([current_a, previous_b], dim=1)
            scores = torch.cat([a_scores, b_scores], dim=1)
            valid = None
            if mask is not None or previous_mask is not None:
                if mask is None:
                    mask = torch.ones(B, a_len, device=current_a.device, dtype=torch.bool)
                if previous_mask is None:
                    previous_mask = torch.ones(B, b_len, device=current_a.device, dtype=torch.bool)
                valid = torch.cat([mask, previous_mask.to(device=current_a.device)], dim=1)

        weights, block_valid = compressor._safe_temporal_softmax(scores=scores, valid=valid, dim=1)
        comp = (weights * tokens).sum(dim=1)
        return torch.where(block_valid[:, None], comp, torch.zeros_like(comp))

    def compress_csa_main_block_for_cache(self, current_a, previous_b, mask, **kwargs):
        return self._compress_csa_block_for_cache(
            self.kv_compressor,
            current_a,
            previous_b,
            mask,
            **kwargs,
        )

    def compress_csa_index_block_for_cache(self, current_a, previous_b, mask, **kwargs):
        return self._compress_csa_block_for_cache(
            self.index_compressor,
            current_a,
            previous_b,
            mask,
            **kwargs,
        )

    def _shape_q(self, q: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.shape
        return q.view(B, T, self.n_heads, self.head_dim)

    def _shape_index_q(self, index_q: torch.Tensor) -> torch.Tensor:
        B, T, _ = index_q.shape
        return index_q.view(B, T, self.n_indexer_heads, self.indexer_dim)

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must have shape [B,T], "
                f"got {tuple(attention_mask.shape)}"
            )

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        return attention_mask

    def _build_local_allowed_mask(
        self,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_pos = torch.arange(T, device=device)[:, None]
        k_pos = torch.arange(T, device=device)[None, :]

        causal = k_pos <= q_pos
        in_window = (q_pos - k_pos) < self.window_size

        return causal & in_window

    def _gather_selected(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        values:  [B,S,D]
        indices: [B,T,K]
        return:  [B,T,K,D]
        """
        B, S, D = values.shape
        B_i, T, K = indices.shape

        if B_i != B:
            raise ValueError(f"Batch mismatch: values B={B}, indices B={B_i}")

        source = values[:, None, :, :].expand(B, T, S, D)
        idx = indices[..., None].expand(B, T, K, D)

        return torch.gather(source, dim=2, index=idx)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------
        if x.dim() != 3:
            raise ValueError(
                f"CSAAttention expects x [B,T,d_model], got {tuple(x.shape)}"
            )

        B, T, C_model = x.shape

        if C_model != self.d_model:
            raise ValueError(
                f"Expected hidden size {self.d_model}, got {C_model}"
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
        # Shared low-rank query path
        # ----------------------------------------------------
        q_latent = self.q_down_proj(x)          # [B,T,Qc]

        q = self.q_up_proj(q_latent)
        q = self._shape_q(q)                    # [B,T,H,Dh]

        index_q = self.index_q_up_proj(q_latent)
        index_q = self._shape_index_q(index_q)  # [B,T,H_i,I]

        index_weights = self.index_weight_proj(x)  # [B,T,H_i]

        # ----------------------------------------------------
        # KV a/b projections
        # ----------------------------------------------------
        C_a = self.a_kv_proj(x)
        C_b = self.b_kv_proj(x)
        Z_a = self.a_z_proj(x)
        Z_b = self.b_z_proj(x)
        # [B,T,Dh]

        if self.local_kv_proj is not None:
            C_local = self.local_kv_proj(x)      # [B,T,Dh]
        else:
            C_local = C_a                        # backward-compatible fallback

        I_a = self.a_index_kv_proj(x)
        I_b = self.b_index_kv_proj(x)
        IZ_a = self.a_index_z_proj(x)
        IZ_b = self.b_index_z_proj(x)
        # [B,T,I]

        # ----------------------------------------------------
        # RoPE on query
        # ----------------------------------------------------
        if self.rope is not None:
            q = self.rope(
                q,
                position_ids=position_ids,
                start_pos=start_pos,
            )

        # ----------------------------------------------------
        # Overlapped compression: KV and index keys
        # ----------------------------------------------------
        C_comp, comp_valid_mask, comp_position_ids = self.kv_compressor(
            C_a=C_a,
            C_b=C_b,
            Z_a=Z_a,
            Z_b=Z_b,
            attention_mask=attention_mask,
            position_ids=position_ids,
            start_pos=start_pos,
        )

        I_comp, index_valid_mask, _ = self.index_compressor(
            C_a=I_a,
            C_b=I_b,
            Z_a=IZ_a,
            Z_b=IZ_b,
            attention_mask=attention_mask,
            position_ids=position_ids,
            start_pos=start_pos,
        )

        if not torch.equal(comp_valid_mask, index_valid_mask):
            raise RuntimeError(
                "KV compressed valid mask differs from index compressed valid mask."
            )

        S = C_comp.shape[1]

        # ----------------------------------------------------
        # RoPE on compressed global keys and local keys
        # ----------------------------------------------------
        if self.rope is not None:
            K_global_all = C_comp[:, :, None, :]  # [B,S,1,Dh]
            K_global_all = self.rope(
                K_global_all,
                position_ids=comp_position_ids,
                start_pos=0,
            )
            K_global_all = K_global_all[:, :, 0, :]  # [B,S,Dh]

            K_local = C_local[:, :, None, :]  # [B,T,1,Dh]
            K_local = self.rope(
                K_local,
                position_ids=position_ids,
                start_pos=start_pos,
            )
            K_local = K_local[:, :, 0, :]  # [B,T,Dh]
        else:
            K_global_all = C_comp
            K_local = C_local

        V_global_all = C_comp
        V_local = C_local

        # ----------------------------------------------------
        # Lightning indexer top-k selection
        # ----------------------------------------------------
        if need_weights:
            topk_indices, topk_scores, topk_mask, index_scores = self.indexer(
                index_q=index_q,
                index_weights=index_weights,
                I_comp=I_comp,
                comp_valid_mask=comp_valid_mask,
                need_scores=True,
            )
        else:
            topk_indices, topk_scores, topk_mask = self.indexer(
                index_q=index_q,
                index_weights=index_weights,
                I_comp=I_comp,
                comp_valid_mask=comp_valid_mask,
                need_scores=False,
            )
            index_scores = None

        K_eff = topk_indices.shape[-1]

        # ----------------------------------------------------
        # Gather selected global K/V
        # ----------------------------------------------------
        K_selected = self._gather_selected(K_global_all, topk_indices)   # [B,T,K,Dh]
        V_selected = self._gather_selected(V_global_all, topk_indices)   # [B,T,K,Dh]

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------
        q = q  # [B,T,H,Dh]

        scores_parts = []
        allowed_parts = []

        # -------------------------
        # Optional attention sink
        # -------------------------
        if self.use_attention_sink:
            K_sink = self.sink_k.expand(B, -1, -1)  # [B,1,Dh]
            scores_sink = torch.einsum(
                "bthd,bsd->bhts",
                q,
                K_sink,
            ) / math.sqrt(self.head_dim)             # [B,H,T,1]

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
        # Sparse selected global scores
        # -------------------------
        scores_global = torch.einsum(
            "bthd,btkd->bhtk",
            q,
            K_selected,
        ) / math.sqrt(self.head_dim)                 # [B,H,T,K]

        # Canonical default: indexer chooses selected blocks only. It does not
        # bias the core attention logits unless explicitly enabled.
        if self.use_indexer_score_bias:
            scores_global = scores_global + topk_scores[:, None, :, :].to(
                dtype=scores_global.dtype
            )

        global_allowed = topk_mask[:, None, :, :].expand(B, self.n_heads, T, K_eff)
        # [B,H,T,K]

        scores_parts.append(scores_global)
        allowed_parts.append(global_allowed)

        # -------------------------
        # Local exact scores
        # -------------------------
        scores_local = torch.einsum(
            "bthd,bsd->bhts",
            q,
            K_local,
        ) / math.sqrt(self.head_dim)                 # [B,H,T,T]

        local_allowed = self._build_local_allowed_mask(
            T=T,
            device=x.device,
        )
        local_allowed = local_allowed[None, None, :, :]  # [1,1,T,T]

        if attention_mask is not None:
            local_key_allowed = attention_mask[:, None, None, :].to(
                device=x.device,
                dtype=torch.bool,
            )
            local_allowed = local_allowed & local_key_allowed

        local_allowed = local_allowed.expand(B, self.n_heads, T, T)

        scores_parts.append(scores_local)
        allowed_parts.append(local_allowed)

        # ----------------------------------------------------
        # Combined sink + sparse global + local softmax
        # ----------------------------------------------------
        scores = torch.cat(scores_parts, dim=-1)          # [B,H,T,N]
        allowed = torch.cat(allowed_parts, dim=-1)        # [B,H,T,N]

        weights = safe_masked_softmax(
            scores=scores,
            allowed_mask=allowed,
            dim=-1,
        )

        weights = self.attention_dropout(weights)

        # ----------------------------------------------------
        # Split attention weights
        # ----------------------------------------------------
        offset = 0

        if self.use_attention_sink:
            weights_sink = weights[..., offset:offset + 1]  # [B,H,T,1]
            offset += 1
        else:
            weights_sink = None

        weights_global = weights[..., offset:offset + K_eff]  # [B,H,T,K]
        offset += K_eff

        weights_local = weights[..., offset:]                 # [B,H,T,T]

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
            V_sink = self.sink_v.expand(B, -1, -1)  # [B,1,Dh]
            context_sink = torch.einsum(
                "bhts,bsd->bhtd",
                weights_sink,
                V_sink,
            )
            context = context + context_sink

        context_global = torch.einsum(
            "bhtk,btkd->bhtd",
            weights_global,
            V_selected,
        )

        context_local = torch.einsum(
            "bhts,bsd->bhtd",
            weights_local,
            V_local,
        )

        context = context + context_global + context_local  # [B,H,T,Dh]

        # ----------------------------------------------------
        # Merge heads + output projection
        # ----------------------------------------------------
        context = context.transpose(1, 2).contiguous()  # [B,T,H,Dh]

        if self.use_grouped_output_projection:
            out = self.out_proj(context)               # [B,T,D]
        else:
            context = context.view(B, T, self.inner_dim)
            out = self.out_proj(context)               # [B,T,D]

        out = self.residual_dropout(out)

        if need_weights:
            aux = {
                "global_attn_weights": weights_global,
                "local_attn_weights": weights_local,
                "topk_indices": topk_indices,
                "topk_scores": topk_scores,
                "topk_mask": topk_mask,
                "compressed_valid_mask": comp_valid_mask,
                "compressed_position_ids": comp_position_ids,
                "index_scores": index_scores,
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
            raise ValueError(f"CSA forward_decode expects x_t [B,1,D], got {tuple(x_t.shape)}")

        B, T, C_model = x_t.shape
        if C_model != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {C_model}")

        if position_ids is None:
            position_ids = torch.full(
                (B, 1),
                int(cache.tokens_seen),
                device=x_t.device,
                dtype=torch.long,
            )
        else:
            position_ids = position_ids.to(device=x_t.device, dtype=torch.long)

        q_latent = self.q_down_proj(x_t)
        q = self._shape_q(self.q_up_proj(q_latent))
        index_q = self._shape_index_q(self.index_q_up_proj(q_latent))
        index_weights = self.index_weight_proj(x_t)

        C_a = self.a_kv_proj(x_t)
        C_b = self.b_kv_proj(x_t)
        Z_a = self.a_z_proj(x_t)
        Z_b = self.b_z_proj(x_t)
        C_local = self.local_kv_proj(x_t) if self.local_kv_proj is not None else C_a

        I_a = self.a_index_kv_proj(x_t)
        I_b = self.b_index_kv_proj(x_t)
        IZ_a = self.a_index_z_proj(x_t)
        IZ_b = self.b_index_z_proj(x_t)

        if self.rope is not None:
            q = self.rope(q, position_ids=position_ids, start_pos=0)

        valid_t = (
            attention_mask[:, -1:].to(device=x_t.device).bool()
            if attention_mask is not None
            else torch.ones(B, 1, device=x_t.device, dtype=torch.bool)
        )

        def real_overlapped_compressor(
            current_a,
            previous_b,
            mask,
            *,
            current_z=None,
            previous_z=None,
            previous_mask=None,
            compressor=None,
        ):
            compressor = compressor or self.kv_compressor
            current_z = current_z if current_z is not None else current_a
            if previous_b is None:
                tokens = current_a
                scores = current_z + compressor.bias_a[: current_a.shape[1], :].to(
                    device=current_a.device,
                    dtype=current_z.dtype,
                )[None, :, :]
                valid = mask
            else:
                previous_z = previous_z if previous_z is not None else previous_b
                a_len = current_a.shape[1]
                b_len = previous_b.shape[1]
                a_scores = current_z + compressor.bias_a[:a_len, :].to(
                    device=current_a.device,
                    dtype=current_z.dtype,
                )[None, :, :]
                b_scores = previous_z + compressor.bias_b[:b_len, :].to(
                    device=previous_b.device,
                    dtype=previous_z.dtype,
                )[None, :, :]
                tokens = torch.cat([current_a, previous_b], dim=1)
                scores = torch.cat([a_scores, b_scores], dim=1)
                valid = None
                if mask is not None or previous_mask is not None:
                    if mask is None:
                        mask = torch.ones(B, a_len, device=x_t.device, dtype=torch.bool)
                    if previous_mask is None:
                        previous_mask = torch.ones(B, b_len, device=x_t.device, dtype=torch.bool)
                    valid = torch.cat([mask, previous_mask.to(device=x_t.device)], dim=1)

            weights, block_valid = compressor._safe_temporal_softmax(scores=scores, valid=valid, dim=1)
            comp = (weights * tokens).sum(dim=1)
            return torch.where(block_valid[:, None], comp, torch.zeros_like(comp))

        def main_compressor(current_a, previous_b, mask, **kwargs):
            return real_overlapped_compressor(
                current_a,
                previous_b,
                mask,
                compressor=self.kv_compressor,
                **kwargs,
            )

        def index_compressor(current_a, previous_b, mask, **kwargs):
            return real_overlapped_compressor(
                current_a,
                previous_b,
                mask,
                compressor=self.index_compressor,
                **kwargs,
            )

        cache.append_token_state(
            a_c_t=C_a,
            b_c_t=C_b,
            a_z_t=Z_a,
            b_z_t=Z_b,
            index_a_c_t=I_a,
            index_b_c_t=I_b,
            index_a_z_t=IZ_a,
            index_b_z_t=IZ_b,
            position_t=position_ids,
            valid_mask_t=valid_t,
        )
        cache.local_c[:, -1:, :] = C_local
        cache.flush_ready_blocks(main_compressor=main_compressor, index_compressor=index_compressor)

        if self.rope is not None:
            q_rope = q
        else:
            q_rope = q

        scores_parts = []
        allowed_parts = []
        value_parts = []
        part_names = []

        if self.use_attention_sink:
            K_sink = self.sink_k.expand(B, -1, -1)
            scores_sink = torch.einsum("bthd,bsd->bhts", q_rope, K_sink) / math.sqrt(self.head_dim)
            scores_parts.append(scores_sink)
            allowed_parts.append(torch.ones(B, self.n_heads, 1, 1, device=x_t.device, dtype=torch.bool))
            value_parts.append(self.sink_v.expand(B, -1, -1))
            part_names.append("sink")

        topk_indices = None
        topk_scores = None
        topk_mask = None
        index_scores = None
        if cache.compressed_main is not None and cache.compressed_main.shape[1] > 0:
            S = cache.compressed_main.shape[1]
            raw = torch.einsum("bthi,bsi->bths", index_q, cache.compressed_index)
            raw = F.relu(raw)
            index_scores = (index_weights[..., None] * raw).sum(dim=2)
            allowed_index = cache.compressed_valid_mask[:, None, :].bool()
            masked_scores = index_scores.masked_fill(~allowed_index, torch.finfo(index_scores.dtype).min)
            K_eff = min(self.top_k, S)
            topk_scores, topk_indices = torch.topk(masked_scores, k=K_eff, dim=-1)
            topk_mask = torch.gather(allowed_index.expand(B, 1, S), dim=-1, index=topk_indices)
            topk_scores = torch.where(topk_mask, topk_scores, torch.zeros_like(topk_scores))

            K_global_all = cache.compressed_main
            if self.rope is not None:
                K_global_rope = K_global_all[:, :, None, :]
                K_global_rope = self.rope(
                    K_global_rope,
                    position_ids=cache.compressed_positions,
                    start_pos=0,
                )
                K_global_all = K_global_rope[:, :, 0, :]

            K_selected = self._gather_selected(K_global_all, topk_indices)
            V_selected = self._gather_selected(cache.compressed_main, topk_indices)
            scores_global = torch.einsum("bthd,btkd->bhtk", q_rope, K_selected) / math.sqrt(self.head_dim)
            if self.use_indexer_score_bias:
                scores_global = scores_global + topk_scores[:, None, :, :].to(scores_global.dtype)
            scores_parts.append(scores_global)
            allowed_parts.append(topk_mask[:, None, :, :].expand(B, self.n_heads, 1, K_eff))
            value_parts.append(V_selected)
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
            scores_local = torch.einsum("bthd,bsd->bhts", q_rope, K_local) / math.sqrt(self.head_dim)
            scores_parts.append(scores_local)
            allowed_parts.append(cache.local_valid_mask[:, None, None, :].expand(B, self.n_heads, 1, cache.local_c.shape[1]))
            value_parts.append(cache.local_c)
            part_names.append("local")

        scores = torch.cat(scores_parts, dim=-1)
        allowed = torch.cat(allowed_parts, dim=-1)
        weights = safe_masked_softmax(scores=scores, allowed_mask=allowed, dim=-1)
        weights = self.attention_dropout(weights)

        context = torch.zeros(B, self.n_heads, 1, self.head_dim, device=x_t.device, dtype=x_t.dtype)
        offset = 0
        aux = {}
        for name, values in zip(part_names, value_parts):
            if name == "global":
                width = values.shape[2]
                part_weights = weights[..., offset : offset + width]
                context = context + torch.einsum("bhtk,btkd->bhtd", part_weights, values)
            else:
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
            aux.update(
                {
                    "topk_indices": topk_indices,
                    "topk_scores": topk_scores,
                    "topk_mask": topk_mask,
                    "compressed_valid_mask": cache.compressed_valid_mask,
                    "compressed_position_ids": cache.compressed_positions,
                    "index_scores": index_scores,
                }
            )
        return out, cache, aux
