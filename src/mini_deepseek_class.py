# ============================================================
# DEEPSEEK V4 LM
# ============================================================

from src.deepseek_block import * 
from src.transformer_modules.embedding_module import * 

def _get_token_embedding_weight(embedding: nn.Module) -> nn.Parameter:
    """
    Supports both:
        TokenEmbedding.weight
    and common internal names:
        TokenEmbedding.embedding.weight
        TokenEmbedding.token_embedding.weight
    """
    if hasattr(embedding, "weight"):
        return embedding.weight

    if hasattr(embedding, "embedding") and hasattr(embedding.embedding, "weight"):
        return embedding.embedding.weight

    if hasattr(embedding, "token_embedding") and hasattr(embedding.token_embedding, "weight"):
        return embedding.token_embedding.weight

    raise AttributeError(
        "Could not find token embedding weight. Expected embedding.weight, "
        "embedding.embedding.weight, or embedding.token_embedding.weight."
    )



class DeepSeekV4LM(nn.Module):
    """
    Configurable DeepSeek-V4-style language model.

    Components:
        TokenEmbedding
        DeepSeekV4Block x n_layers
        final RMSNorm
        LM head
        optional MTP head

    Supported switches:
        attention_type: "mha", "hca", "csa", "hybrid"
        attention_pattern: cyclic per-layer schedule used when attention_type="hybrid"
        ffn_type: "dense", "moe"
        use_mhc: bool
        use_mtp: bool
    """

    def __init__(self, config: DeepSeekV4LMConfig):
        super().__init__()

        config.validate()

        self.config = config

        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len
        self.pad_token_id = config.pad_token_id
        self.ignore_index = config.ignore_index
        self.labels_are_shifted = config.labels_are_shifted
        self.ignore_pad_token_in_loss = config.ignore_pad_token_in_loss
        self.use_mhc = config.use_mhc
        self.use_mtp = config.use_mtp

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------
        self.embedding = TokenEmbedding(
            EmbeddingConfig(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                pad_token_id=config.pad_token_id,
                max_seq_len=config.max_seq_len,
                embedding_dropout=config.embedding_dropout,
                scale_embeddings=config.scale_embeddings,
                init_std=config.init_std,
                tie_word_embeddings=config.tie_word_embeddings,
            )
        )

        # ----------------------------------------------------
        # Blocks
        # ----------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                DeepSeekV4Block(config=config, layer_idx=i)
                for i in range(config.n_layers)
            ]
        )

        # ----------------------------------------------------
        # mHC readout
        # ----------------------------------------------------
        if config.use_mhc and config.mhc_collapse_mode == "readout":
            self.mhc_readout = HyperConnectionReadout(
                n_hc=config.n_hc,
                init="mean",
            )
        else:
            self.mhc_readout = None

        # ----------------------------------------------------
        # Final norm + LM head
        # ----------------------------------------------------
        self.final_norm = RMSNorm(
            dim=config.d_model,
            eps=config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.reset_lm_head_parameters()

        if config.tie_word_embeddings:
            self.tie_lm_head_to_embeddings()

        # ----------------------------------------------------
        # Optional MTP
        # ----------------------------------------------------
        if config.use_mtp:
            self.mtp_head = MultiTokenPredictionHead(
                MTPConfig(
                    d_model=config.d_model,
                    vocab_size=config.vocab_size,
                    mtp_depth=config.mtp_depth,
                    hidden_dim=config.mtp_hidden_dim,
                    use_mtp_transform=config.use_mtp_transform,
                    activation=config.mtp_activation,
                    dropout=config.mtp_dropout,
                    use_bias=config.use_mlp_bias,
                    init_std=config.init_std,
                    tie_with_lm_head=config.mtp_tie_with_lm_head,
                    mtp_loss_weight=config.mtp_loss_weight,
                    ignore_index=config.ignore_index,
                    pad_token_id=config.pad_token_id,
                    depth_loss_weights=config.mtp_depth_loss_weights,
                    validate_label_range=config.mtp_validate_label_range,
                )
            )

            if config.mtp_tie_with_lm_head:
                self.mtp_head.tie_weights(self.lm_head.weight)
        else:
            self.mtp_head = None

    def reset_lm_head_parameters(self) -> None:
        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=self.config.init_std)

    def tie_lm_head_to_embeddings(self) -> None:
        self.lm_head.weight = _get_token_embedding_weight(self.embedding)

    def _validate_input_ids(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape [B,T], got {tuple(input_ids.shape)}")

        if torch.is_floating_point(input_ids):
            raise TypeError("input_ids must be integer token ids, not floating point.")

        B, T = input_ids.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}")

        return B, T

    def _validate_labels(self, labels: torch.Tensor, input_ids: torch.Tensor) -> None:
        if labels.shape != input_ids.shape:
            raise ValueError(
                f"labels must have shape {tuple(input_ids.shape)}, "
                f"got {tuple(labels.shape)}")

        if torch.is_floating_point(labels):
            raise TypeError("labels must be integer token ids, not floating point.")

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor) -> torch.Tensor:

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids. "
                f"Expected {tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}")

        return attention_mask

    def _build_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:

        if attention_mask is not None:
            return self._validate_attention_mask(attention_mask, input_ids)

        if self.pad_token_id is None:
            return None

        return (input_ids != self.pad_token_id).long()
    def _prepare_lm_loss_inputs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare logits and labels for causal LM loss.

        Supports two conventions:

        1. labels_are_shifted=True

            input_ids: [x_0, x_1, ..., x_{T-1}]
            labels:    [x_1, x_2, ..., x_T]

            The dataset already provides next-token labels.
            We compute CE over all positions:

                logits[:, :, :] vs labels[:, :]

        2. labels_are_shifted=False

            input_ids: [x_0, x_1, ..., x_{T-1}]
            labels:    [x_0, x_1, ..., x_{T-1}]

            HuggingFace-style labels=input_ids.
            We shift internally:

                logits[:, :-1, :] vs labels[:, 1:]

        Padding handling:
            If ignore_pad_token_in_loss=True and pad_token_id is not None,
            target labels equal to pad_token_id are converted to ignore_index.

            If attention_mask is provided, target positions with mask == 0 are
            also converted to ignore_index.
        """

        if logits.dim() != 3:
            raise ValueError(
                f"logits must have shape [B,T,V], got {tuple(logits.shape)}"
            )

        if labels.dim() != 2:
            raise ValueError(
                f"labels must have shape [B,T], got {tuple(labels.shape)}"
            )

        B, T, V = logits.shape

        if labels.shape != (B, T):
            raise ValueError(
                f"labels must have shape {(B, T)}, got {tuple(labels.shape)}"
            )

        if torch.is_floating_point(labels):
            raise TypeError("labels must be integer token ids, not floating point.")

        if attention_mask is not None and attention_mask.shape != labels.shape:
            raise ValueError(
                "attention_mask must have same shape as labels. "
                f"Expected {tuple(labels.shape)}, got {tuple(attention_mask.shape)}"
            )

        labels = labels.long()

        if self.labels_are_shifted:
            loss_logits = logits
            loss_labels = labels
            loss_attention_mask = attention_mask
        else:
            if T < 2:
                raise ValueError(
                    "Cannot compute internally shifted LM loss with sequence length < 2."
                )

            loss_logits = logits[:, :-1, :].contiguous()
            loss_labels = labels[:, 1:].contiguous()
            loss_attention_mask = (
                attention_mask[:, 1:].contiguous()
                if attention_mask is not None
                else None
            )

        loss_labels = loss_labels.clone()

        # Ignore pad-token targets.
        if self.ignore_pad_token_in_loss and self.pad_token_id is not None:
            loss_labels = torch.where(
                loss_labels == int(self.pad_token_id),
                torch.full_like(loss_labels, int(self.ignore_index)),
                loss_labels,
            )

        # Ignore masked target positions.
        if loss_attention_mask is not None:
            loss_labels = torch.where(
                loss_attention_mask.to(device=loss_labels.device, dtype=torch.bool),
                loss_labels,
                torch.full_like(loss_labels, int(self.ignore_index)),
            )

        return loss_logits, loss_labels

    def _compute_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute causal LM loss under either shifted-label or HF-style convention.
        """

        loss_logits, loss_labels = self._prepare_lm_loss_inputs(
            logits=logits,
            labels=labels,
            attention_mask=attention_mask,
        )

        B, T_loss, V = loss_logits.shape

        return F.cross_entropy(
            loss_logits.reshape(B * T_loss, V),
            loss_labels.reshape(B * T_loss),
            ignore_index=self.ignore_index,
        )

    def _collect_moe_aux_loss(
        self,
        block_aux_list: list,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:

        """
        Sum MoE auxiliary losses across blocks.

        Supports canonical keys:
            total_balance_loss
            balance_loss
            sequence_balance_loss

        If no MoE losses are present, returns scalar zero.
        """
        total = torch.zeros((), device=device, dtype=dtype)

        for block_aux in block_aux_list:
            if not isinstance(block_aux, dict):
                continue

            moe_aux = block_aux.get("moe", None)
            if not isinstance(moe_aux, dict):
                continue

            if "total_balance_loss" in moe_aux and moe_aux["total_balance_loss"] is not None:
                total = total + moe_aux["total_balance_loss"].to(device=device, dtype=dtype)
                continue

            if "balance_loss" in moe_aux and moe_aux["balance_loss"] is not None:
                total = total + moe_aux["balance_loss"].to(device=device, dtype=dtype)

            if "sequence_balance_loss" in moe_aux and moe_aux["sequence_balance_loss"] is not None:
                total = total + moe_aux["sequence_balance_loss"].to(device=device, dtype=dtype)

        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mtp_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        return_aux: bool = False,
        need_weights: bool = False,
        cache_builder: Optional[Any] = None) -> Dict[str, Any]:


        B, T = self._validate_input_ids(input_ids)

        input_ids = input_ids.long()

        if labels is not None:
            self._validate_labels(labels, input_ids)
            labels = labels.long()

        attention_mask = self._build_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask)

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------
        x = self.embedding(input_ids)
        # [B,T,D]

        block_aux_list = []

        # ----------------------------------------------------
        # Body
        # ----------------------------------------------------
        if self.use_mhc:
            X = expand_residual_stream(
                x,
                n_hc=self.config.n_hc,
                mode=self.config.mhc_expand_mode,
            )

            for block in self.blocks:
                if return_aux or need_weights or self.config.ffn_type == "moe":
                    X, block_aux = block(
                        X,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        start_pos=start_pos,
                        input_ids=input_ids,
                        return_aux=return_aux,
                        need_weights=need_weights,
                        collect_moe_aux=self.config.ffn_type == "moe",
                        cache_builder=cache_builder,
                    )
                    block_aux_list.append(block_aux)
                else:
                    X = block(
                        X,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        start_pos=start_pos,
                        input_ids=input_ids,
                        return_aux=False,
                        need_weights=False,
                        collect_moe_aux=False,
                        cache_builder=cache_builder,
                    )

            if self.config.mhc_collapse_mode == "readout":
                x = self.mhc_readout(X)
            else:
                x = collapse_residual_stream(
                    X,
                    mode=self.config.mhc_collapse_mode,
                )

        else:
            for block in self.blocks:
                if return_aux or need_weights or self.config.ffn_type == "moe":
                    x, block_aux = block(
                        x,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        start_pos=start_pos,
                        input_ids=input_ids,
                        return_aux=return_aux,
                        need_weights=need_weights,
                        collect_moe_aux=self.config.ffn_type == "moe",
                        cache_builder=cache_builder,
                    )
                    block_aux_list.append(block_aux)
                else:
                    x = block(
                        x,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        start_pos=start_pos,
                        input_ids=input_ids,
                        return_aux=False,
                        need_weights=False,
                        collect_moe_aux=False,
                        cache_builder=cache_builder,
                    )

        # ----------------------------------------------------
        # Final norm + logits
        # ----------------------------------------------------
        hidden_states = self.final_norm(x)
        logits = self.lm_head(hidden_states)

        # ----------------------------------------------------
        # LM loss
        # ----------------------------------------------------
        lm_loss = None

        if labels is not None:
            lm_loss = self._compute_lm_loss(
                logits=logits,
                labels=labels,
                attention_mask=attention_mask,
            )

        # ----------------------------------------------------
        # MTP
        # ----------------------------------------------------
        mtp_loss = None
        mtp_outputs = None

        if self.use_mtp:
            if mtp_labels is None and labels is not None:
                mtp_labels = build_mtp_labels(
                    input_ids=input_ids,
                    mtp_depth=self.config.mtp_depth,
                    ignore_index=self.ignore_index,
                    pad_token_id=self.pad_token_id,
                )

            mtp_outputs = self.mtp_head(
                hidden_states,
                mtp_labels=mtp_labels,
                return_aux=return_aux)

            mtp_loss = mtp_outputs["mtp_loss"]

        # ----------------------------------------------------
        # MoE aux loss
        # ----------------------------------------------------
        moe_aux_loss = self._collect_moe_aux_loss(
            block_aux_list=block_aux_list,
            device=logits.device,
            dtype=logits.dtype)

        has_moe_aux_loss = bool(
            self.config.ffn_type == "moe"
            and (
                self.config.balance_loss_weight > 0
                or self.config.sequence_balance_loss_weight > 0
            ))

        if not has_moe_aux_loss:
            # Keep output clean: no loss contribution unless explicitly weighted.
            moe_aux_loss = None

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------
        loss = None

        if lm_loss is not None:
            loss = lm_loss

            if mtp_loss is not None:
                loss = loss + mtp_loss.to(dtype=loss.dtype)

            if moe_aux_loss is not None:
                loss = loss + moe_aux_loss.to(dtype=loss.dtype)

        # ----------------------------------------------------
        # Aux
        # ----------------------------------------------------
        aux: Dict[str, Any] = {}

        if return_aux or need_weights:
          aux["blocks"] = block_aux_list
          aux["labels_are_shifted"] = self.labels_are_shifted
          aux["ignore_pad_token_in_loss"] = self.ignore_pad_token_in_loss

          if mtp_outputs is not None:
              aux["mtp"] = mtp_outputs.get("aux", {})

          if attention_mask is not None:
              aux["attention_mask"] = attention_mask

        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "mtp_loss": mtp_loss,
            "moe_aux_loss": moe_aux_loss,
            "hidden_states": hidden_states if return_aux else None,
            "aux": aux}

    @torch.no_grad()
    def prefill_decode_cache(
        self,
        input_ids: torch.Tensor,
        cache,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inference_config: Optional[Any] = None,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        from inference.deepseek_cache_builder import DeepSeekActiveCacheBuilder

        del inference_config
        B, T = self._validate_input_ids(input_ids)
        input_ids = input_ids.to(device=cache.device, dtype=torch.long)
        if position_ids is None:
            position_ids = torch.arange(T, device=cache.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        else:
            position_ids = position_ids.to(device=cache.device, dtype=torch.long)
        attention_mask = self._build_attention_mask(input_ids=input_ids, attention_mask=attention_mask)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=cache.device)

        builder = DeepSeekActiveCacheBuilder(cache=cache, inference_config=None)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_aux=return_aux,
            need_weights=False,
            cache_builder=builder,
        )
        cache.sequence_ids = input_ids.detach()
        cache.attention_mask = attention_mask.detach() if attention_mask is not None else None
        cache.tokens_seen = int(T)
        return {
            "logits": outputs["logits"][:, -1:, :],
            "full_logits": outputs["logits"],
            "hidden_states": outputs.get("hidden_states", None),
            "cache": cache,
            "aux": outputs.get("aux", {}) if return_aux else {},
        }

    def forward_decode(
        self,
        input_ids_t: torch.Tensor,
        cache,
        position_ids_t: Optional[torch.Tensor] = None,
        attention_mask_t: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        if input_ids_t.dim() != 2 or input_ids_t.shape[1] != 1:
            raise ValueError(f"input_ids_t must have shape [B,1], got {tuple(input_ids_t.shape)}")

        input_ids_t = input_ids_t.to(device=cache.device, dtype=torch.long)
        B, _ = input_ids_t.shape
        if B != cache.batch_size:
            raise ValueError(f"cache batch_size={cache.batch_size} does not match input batch size {B}")

        if position_ids_t is None:
            position_ids_t = torch.full(
                (B, 1),
                int(cache.tokens_seen),
                device=cache.device,
                dtype=torch.long,
            )
        else:
            position_ids_t = position_ids_t.to(device=cache.device, dtype=torch.long)

        if attention_mask_t is None:
            if self.pad_token_id is None:
                attention_mask_t = torch.ones(B, 1, device=cache.device, dtype=torch.long)
            else:
                attention_mask_t = input_ids_t.ne(int(self.pad_token_id)).long()
        else:
            attention_mask_t = attention_mask_t.to(device=cache.device)

        full_attention_mask = (
            attention_mask_t
            if cache.attention_mask is None
            else torch.cat([cache.attention_mask.to(device=cache.device), attention_mask_t], dim=1)
        )

        x_t = self.embedding(input_ids_t)
        if self.use_mhc:
            x_or_X = expand_residual_stream(
                x_t,
                n_hc=self.config.n_hc,
                mode=self.config.mhc_expand_mode,
            )
        else:
            x_or_X = x_t
        block_aux_list = []

        for layer_idx, block in enumerate(self.blocks):
            x_or_X, new_layer_cache, block_aux = block.forward_decode(
                x_or_X,
                layer_cache=cache.layer_caches[layer_idx],
                attention_mask_t=full_attention_mask,
                position_ids_t=position_ids_t,
                input_ids_t=input_ids_t,
                return_aux=return_aux,
            )
            cache.layer_caches[layer_idx] = new_layer_cache
            if return_aux or self.config.ffn_type == "moe":
                block_aux_list.append(block_aux)

        if self.use_mhc:
            if self.config.mhc_collapse_mode == "readout":
                x_t = self.mhc_readout(x_or_X)
            else:
                x_t = collapse_residual_stream(
                    x_or_X,
                    mode=self.config.mhc_collapse_mode,
                )
        else:
            x_t = x_or_X

        hidden_states = self.final_norm(x_t)
        logits = self.lm_head(hidden_states)
        cache.append_input_ids(input_ids_t, attention_mask_t)

        aux: Dict[str, Any] = {}
        if return_aux:
            aux["blocks"] = block_aux_list
            aux["cache_summary"] = cache.cache_summary()

        return {
            "logits": logits,
            "hidden_states": hidden_states if return_aux else None,
            "cache": cache,
            "aux": aux,
        }
