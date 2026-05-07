# ============================================================
# DEEPSEEK V4 BLOCK
# ============================================================

from src.deepseek_components import * 

def _supports_kwarg(module_or_fn, name: str) -> bool:
    try:
        sig = inspect.signature(module_or_fn.forward if isinstance(module_or_fn, nn.Module) else module_or_fn)
        return name in sig.parameters
    except Exception:
        return False
    

class DeepSeekV4Block(nn.Module):
    """
    Configurable DeepSeek-V4-style block.

    If use_mhc=False:
        x -> x + attention(norm1(x))
        x -> x + ffn(norm2(x))

    If use_mhc=True:
        X: [B,T,n_hc,D]
        X -> mHC attention update
        X -> mHC FFN update

    This block is intentionally compatibility-oriented:
        - MHA may not support need_weights.
        - HCA/CSA usually support need_weights.
        - Dense FFN has no aux.
        - MoE may need aux collection even when return_aux=False,
          because aux losses can contribute to the final LM loss.
        - Hash MoE needs input_ids.
    """

    def __init__(self, config: DeepSeekV4LMConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.d_model = config.d_model
        self.use_mhc = config.use_mhc
        self.ffn_type = config.ffn_type
        self.attention_type = get_layer_attention_type(config, layer_idx)

        self.norm1 = RMSNorm(
            dim=config.d_model,
            eps=config.rms_norm_eps,
        )

        self.attention = build_deepseek_attention(
            config=config,
            attention_type=self.attention_type,
        )

        self.norm2 = RMSNorm(
            dim=config.d_model,
            eps=config.rms_norm_eps,
        )

        self.ffn = build_deepseek_ffn(config)

        if config.use_mhc:
            self.mhc_attn = build_deepseek_mhc(config)
            self.mhc_ffn = build_deepseek_mhc(config)
        else:
            self.mhc_attn = None
            self.mhc_ffn = None

    # --------------------------------------------------------
    # Attention call helper
    # --------------------------------------------------------

    def _call_attention(
        self,
        x_norm: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        start_pos: int,
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Calls attention in a compatibility-safe way.

        Some modules, such as HCA/CSA, support:
            need_weights=True

        A simpler MHA module may not. Therefore this helper only passes
        arguments that the module actually supports.
        """

        kwargs = {}

        if _supports_kwarg(self.attention, "attention_mask"):
            kwargs["attention_mask"] = attention_mask

        if _supports_kwarg(self.attention, "position_ids"):
            kwargs["position_ids"] = position_ids

        if _supports_kwarg(self.attention, "start_pos"):
            kwargs["start_pos"] = start_pos

        if _supports_kwarg(self.attention, "need_weights"):
            kwargs["need_weights"] = need_weights

        result = self.attention(x_norm, **kwargs)

        # HCA/CSA with need_weights=True usually return:
        #   out, aux
        if isinstance(result, tuple):
            if len(result) != 2:
                raise RuntimeError(
                    "Attention module returned a tuple, but expected "
                    "(attention_output, attention_aux)."
                )

            attn_out, attn_aux = result
            return attn_out, attn_aux

        # MHA or attention modules without aux.
        return result, None

    # --------------------------------------------------------
    # FFN / MoE call helper
    # --------------------------------------------------------

    def _call_ffn(
        self,
        x_norm: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        collect_aux: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Calls either dense SwiGLU or DeepSeekMoE.

        Dense:
            out = ffn(x)

        MoE:
            out, aux = ffn(x, input_ids=input_ids, return_aux=True/False)

        We may collect MoE aux even when the user did not ask for aux,
        because balance losses can be part of the total training loss.
        """

        if self.ffn_type == "dense":
            return self.ffn(x_norm), None

        if self.ffn_type != "moe":
            raise RuntimeError(f"Unknown ffn_type={self.ffn_type}")

        kwargs = {}

        if _supports_kwarg(self.ffn, "input_ids"):
            kwargs["input_ids"] = input_ids

        if _supports_kwarg(self.ffn, "return_aux"):
            kwargs["return_aux"] = collect_aux

        result = self.ffn(x_norm, **kwargs)

        if isinstance(result, tuple):
            if len(result) != 2:
                raise RuntimeError(
                    "MoE module returned a tuple, but expected "
                    "(ffn_output, moe_aux)."
                )

            ffn_out, ffn_aux = result
            return ffn_out, ffn_aux

        return result, None

    # --------------------------------------------------------
    # mHC helper
    # --------------------------------------------------------

    def _mhc_update(
        self,
        mhc: "ManifoldHyperConnection",
        X: torch.Tensor,
        sublayer_fn,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Uses the canonical modular mHC API when available:

            A, B, C = compute_ABC(X)
            x_sub = pre_mix(X, A=A)
            y_sub = sublayer_fn(x_sub)
            X_next = update(X, y_sub, B_mat=B, C=C)

        Falls back to the wrapper API if needed:

            X_next, aux = mhc(X, sublayer=sublayer_fn, return_aux=True)
        """

        if mhc is None:
            raise RuntimeError("mHC module is None but _mhc_update was called.")

        if all(hasattr(mhc, name) for name in ["compute_ABC", "pre_mix", "update"]):
            A, B_mat, C = mhc.compute_ABC(X)

            x_sub = mhc.pre_mix(X, A=A)

            y_sub = sublayer_fn(x_sub)

            X_next = mhc.update(
                X,
                y_sub,
                B_mat=B_mat,
                C=C,
            )

            aux = {
                "A": A,
                "B": B_mat,
                "C": C,
                "x_sub": x_sub,
                "y_sub": y_sub,
            }

            return X_next, aux

        # Fallback for older mHC wrapper.
        result = mhc(
            X,
            sublayer=sublayer_fn,
            return_aux=True,
        )

        if isinstance(result, tuple):
            if len(result) != 2:
                raise RuntimeError(
                    "mHC returned a tuple, but expected (X_next, aux)."
                )

            X_next, aux = result
            return X_next, aux

        return result, {}

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x_or_X: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        input_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        need_weights: bool = False,
        collect_moe_aux: Optional[bool] = None,
        cache_builder: Optional[Any] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:

        aux: Dict[str, Any] = {}

        # If not explicitly provided, collect MoE aux whenever the block uses MoE.
        # This is useful because the LM wrapper may need balance losses even
        # when return_aux=False.
        if collect_moe_aux is None:
            collect_moe_aux = self.ffn_type == "moe"

        # ====================================================
        # Standard residual path
        # ====================================================
        if not self.use_mhc:
            x = x_or_X

            if x.dim() != 3:
                raise ValueError(
                    "DeepSeekV4Block without mHC expects x [B,T,D], "
                    f"got {tuple(x.shape)}"
                )

            if x.shape[-1] != self.d_model:
                raise ValueError(
                    f"Expected hidden size {self.d_model}, got {x.shape[-1]}"
                )

            # -------------------------
            # Attention residual update
            # -------------------------
            residual = x
            x_norm = self.norm1(x)
            if cache_builder is not None:
                cache_builder.capture_layer_input(
                    layer_idx=self.layer_idx,
                    attention_type=self.attention_type,
                    attention_module=self.attention,
                    x_norm=x_norm,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

            attn_out, attn_aux = self._call_attention(
                x_norm=x_norm,
                attention_mask=attention_mask,
                position_ids=position_ids,
                start_pos=start_pos,
                need_weights=need_weights,
            )

            if attn_out.shape != residual.shape:
                raise ValueError(
                    "Attention output must have same shape as residual. "
                    f"Expected {tuple(residual.shape)}, got {tuple(attn_out.shape)}"
                )

            x = residual + attn_out

            if attn_aux is not None:
                aux["attention"] = attn_aux

            # -------------------------
            # FFN / MoE residual update
            # -------------------------
            residual = x
            x_norm = self.norm2(x)

            ffn_out, ffn_aux = self._call_ffn(
                x_norm=x_norm,
                input_ids=input_ids,
                collect_aux=collect_moe_aux or return_aux,
            )

            if ffn_out.shape != residual.shape:
                raise ValueError(
                    "FFN output must have same shape as residual. "
                    f"Expected {tuple(residual.shape)}, got {tuple(ffn_out.shape)}"
                )

            x = residual + ffn_out

            if ffn_aux is not None:
                aux["moe"] = ffn_aux

            if return_aux or need_weights or (collect_moe_aux and ffn_aux is not None):
                return x, aux

            return x

        # ====================================================
        # mHC residual path
        # ====================================================
        X = x_or_X

        if X.dim() != 4:
            raise ValueError(
                "DeepSeekV4Block with mHC expects X [B,T,n_hc,D], "
                f"got {tuple(X.shape)}"
            )

        if X.shape[2] != self.config.n_hc:
            raise ValueError(
                f"Expected n_hc={self.config.n_hc}, got {X.shape[2]}"
            )

        if X.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected hidden size {self.d_model}, got {X.shape[-1]}"
            )

        # -------------------------
        # mHC attention update
        # -------------------------
        attn_aux_holder: Dict[str, Any] = {}

        def attn_sublayer(x_sub: torch.Tensor) -> torch.Tensor:
            x_norm = self.norm1(x_sub)
            if cache_builder is not None:
                cache_builder.capture_layer_input(
                    layer_idx=self.layer_idx,
                    attention_type=self.attention_type,
                    attention_module=self.attention,
                    x_norm=x_norm,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

            attn_out, attn_aux = self._call_attention(
                x_norm=x_norm,
                attention_mask=attention_mask,
                position_ids=position_ids,
                start_pos=start_pos,
                need_weights=need_weights,
            )

            if attn_aux is not None:
                attn_aux_holder["attention"] = attn_aux

            return attn_out

        X, mhc_attn_aux = self._mhc_update(
            mhc=self.mhc_attn,
            X=X,
            sublayer_fn=attn_sublayer,
        )

        aux["mhc_attn"] = mhc_attn_aux

        if "attention" in attn_aux_holder:
            aux["attention"] = attn_aux_holder["attention"]

        # -------------------------
        # mHC FFN / MoE update
        # -------------------------
        ffn_aux_holder: Dict[str, Any] = {}

        def ffn_sublayer(x_sub: torch.Tensor) -> torch.Tensor:
            x_norm = self.norm2(x_sub)

            ffn_out, ffn_aux = self._call_ffn(
                x_norm=x_norm,
                input_ids=input_ids,
                collect_aux=collect_moe_aux or return_aux,
            )

            if ffn_aux is not None:
                ffn_aux_holder["moe"] = ffn_aux

            return ffn_out

        X, mhc_ffn_aux = self._mhc_update(
            mhc=self.mhc_ffn,
            X=X,
            sublayer_fn=ffn_sublayer,
        )

        aux["mhc_ffn"] = mhc_ffn_aux

        if "moe" in ffn_aux_holder:
            aux["moe"] = ffn_aux_holder["moe"]

        if return_aux or need_weights or (collect_moe_aux and "moe" in aux):
            return X, aux

        return X

    def forward_decode(
        self,
        x_t: torch.Tensor,
        layer_cache,
        attention_mask_t: Optional[torch.Tensor] = None,
        position_ids_t: Optional[torch.Tensor] = None,
        input_ids_t: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if not hasattr(self.attention, "forward_decode"):
            raise NotImplementedError(
                f"Attention module {type(self.attention).__name__} does not implement forward_decode."
            )

        aux: Dict[str, Any] = {}

        if self.use_mhc:
            X_t = x_t
            if X_t.dim() != 4 or X_t.shape[1] != 1:
                raise ValueError(f"mHC forward_decode expects X_t [B,1,n_hc,D], got {tuple(X_t.shape)}")

            attn_aux_holder: Dict[str, Any] = {}
            layer_cache_holder = {"cache": layer_cache}

            def attn_sublayer(x_sub: torch.Tensor) -> torch.Tensor:
                x_norm = self.norm1(x_sub)
                attn_out, new_cache, attn_aux = self.attention.forward_decode(
                    x_norm,
                    cache=layer_cache_holder["cache"],
                    position_ids=position_ids_t,
                    attention_mask=attention_mask_t,
                    need_weights=return_aux,
                )
                layer_cache_holder["cache"] = new_cache
                if attn_aux:
                    attn_aux_holder["attention"] = attn_aux
                return attn_out

            X_t, mhc_attn_aux = self._mhc_update(
                mhc=self.mhc_attn,
                X=X_t,
                sublayer_fn=attn_sublayer,
            )
            aux["mhc_attn"] = mhc_attn_aux
            if "attention" in attn_aux_holder:
                aux["attention"] = attn_aux_holder["attention"]

            ffn_aux_holder: Dict[str, Any] = {}

            def ffn_sublayer(x_sub: torch.Tensor) -> torch.Tensor:
                x_norm = self.norm2(x_sub)
                ffn_out, ffn_aux = self._call_ffn(
                    x_norm=x_norm,
                    input_ids=input_ids_t,
                    collect_aux=return_aux or self.ffn_type == "moe",
                )
                if ffn_aux is not None:
                    ffn_aux_holder["moe"] = ffn_aux
                return ffn_out

            X_t, mhc_ffn_aux = self._mhc_update(
                mhc=self.mhc_ffn,
                X=X_t,
                sublayer_fn=ffn_sublayer,
            )
            aux["mhc_ffn"] = mhc_ffn_aux
            if "moe" in ffn_aux_holder:
                aux["moe"] = ffn_aux_holder["moe"]

            return X_t, layer_cache_holder["cache"], aux

        if x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(f"forward_decode expects x_t [B,1,D], got {tuple(x_t.shape)}")

        residual = x_t
        x_norm = self.norm1(x_t)
        attn_out, layer_cache, attn_aux = self.attention.forward_decode(
            x_norm,
            cache=layer_cache,
            position_ids=position_ids_t,
            attention_mask=attention_mask_t,
            need_weights=return_aux,
        )
        x_t = residual + attn_out
        if attn_aux:
            aux["attention"] = attn_aux

        residual = x_t
        x_norm = self.norm2(x_t)
        ffn_out, ffn_aux = self._call_ffn(
            x_norm=x_norm,
            input_ids=input_ids_t,
            collect_aux=return_aux or self.ffn_type == "moe",
        )
        x_t = residual + ffn_out
        if ffn_aux is not None:
            aux["moe"] = ffn_aux

        return x_t, layer_cache, aux
