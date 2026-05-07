from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch

from data.syntethic_long_context_retrieval import SimpleWordTokenizer, SyntheticRetrievalConfig
from inference import inference_autoregresive
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _load_model_config(config_json: Path) -> DeepSeekV4LMConfig:
    meta = json.loads(config_json.read_text(encoding="utf-8"))
    cfg_dict = dict(meta["config"] if "config" in meta else meta)
    if isinstance(cfg_dict.get("attention_pattern"), list):
        cfg_dict["attention_pattern"] = tuple(cfg_dict["attention_pattern"])
    if isinstance(cfg_dict.get("mtp_depth_loss_weights"), list):
        cfg_dict["mtp_depth_loss_weights"] = tuple(cfg_dict["mtp_depth_loss_weights"])
    return DeepSeekV4LMConfig(**cfg_dict)


def _load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError(
        "Unsupported checkpoint format. Expected a state dict or a dict with 'model_state_dict'."
    )


def load_model(checkpoint_path: Path, config_json: Path, device: str) -> DeepSeekV4LM:
    config = _load_model_config(config_json)
    model = DeepSeekV4LM(config)
    model.load_state_dict(_load_checkpoint_state(checkpoint_path))
    if device == "auto":
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target = torch.device(device)
    model.to(target)
    model.eval()
    return model


def build_synthetic_tokenizer(args: argparse.Namespace) -> SimpleWordTokenizer:
    cfg = SyntheticRetrievalConfig(
        num_key_types=args.synthetic_num_key_types,
        num_value_types=args.synthetic_num_value_types,
        vocab_filler_size=args.synthetic_vocab_filler_size,
        num_keys_per_example=args.synthetic_num_keys_per_example,
        block_size=args.synthetic_block_size,
    )
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(cfg)
    return tokenizer


def load_tokenizer(args: argparse.Namespace) -> Optional[Any]:
    if args.tokenizer_path is not None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency branch.
            raise ImportError(
                "Loading tokenizer JSON files requires tokenizers. "
                'Install with: pip install -e ".[data]"'
            ) from exc
        return Tokenizer.from_file(str(args.tokenizer_path))
    if args.synthetic_tokenizer:
        return build_synthetic_tokenizer(args)
    return None


def parse_prompt(args: argparse.Namespace) -> str | list[int]:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_ids is not None:
        return [int(part.strip()) for part in args.prompt_ids.split(",") if part.strip()]
    raise ValueError("Provide either --prompt or --prompt-ids.")


def cmd_generate(args: argparse.Namespace) -> None:
    model = load_model(
        checkpoint_path=Path(args.checkpoint),
        config_json=Path(args.config_json),
        device=args.device,
    )
    tokenizer = load_tokenizer(args)
    prompt = parse_prompt(args)

    out = inference_autoregresive(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        cache_mode=args.cache_mode,
        cache_dtype=args.cache_dtype,
        device=args.device,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
        deepseek_prefill_mode=args.deepseek_prefill_mode,
        local_window_size=args.local_window_size,
        max_cache_length=args.max_cache_length,
        use_mtp_draft=args.use_mtp_draft,
        mtp_accept_mode=args.mtp_accept_mode,
        max_mtp_draft_tokens=args.max_mtp_draft_tokens,
        return_cache_stats=args.return_cache_stats,
    )

    result = {
        "text": out.get("text"),
        "sequences": out["sequences"].detach().cpu().tolist(),
        "num_generated_tokens": out["num_generated_tokens"],
        "prompt_length": out["prompt_length"],
        "cache_stats": out.get("cache_stats"),
        "speed": out.get("speed"),
    }
    if args.include_scores and out.get("scores") is not None:
        result["num_score_tensors"] = len(out["scores"])
    if args.use_mtp_draft and out.get("mtp_drafts") is not None:
        result["mtp_drafts"] = [
            {
                "draft_token_ids": draft["draft_token_ids"].detach().cpu().tolist(),
                "is_speculative_decode": draft["is_speculative_decode"],
            }
            for draft in out["mtp_drafts"]
        ]
    _print_json(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with DeepSeek-V4 Mini inference caches")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Load a checkpoint and generate text")
    generate.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    generate.add_argument("--config-json", required=True, help="Path to model config JSON")
    prompt = generate.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Text prompt. Requires --tokenizer-path or --synthetic-tokenizer.")
    prompt.add_argument("--prompt-ids", help="Comma-separated token ids, e.g. 1,4,5,6")
    generate.add_argument("--tokenizer-path", type=Path, default=None)
    generate.add_argument("--synthetic-tokenizer", action="store_true")
    generate.add_argument("--synthetic-num-key-types", type=int, default=64)
    generate.add_argument("--synthetic-num-value-types", type=int, default=64)
    generate.add_argument("--synthetic-vocab-filler-size", type=int, default=68)
    generate.add_argument("--synthetic-num-keys-per-example", type=int, default=8)
    generate.add_argument("--synthetic-block-size", type=int, default=64)

    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--cache-mode", choices=["audit", "mha_decode", "deepseek_decode"], default="deepseek_decode")
    generate.add_argument("--deepseek-prefill-mode", choices=["parallel", "sequential_debug"], default="parallel")
    generate.add_argument("--cache-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    generate.add_argument("--device", default="auto")
    generate.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--top-p", type=float, default=None)
    generate.add_argument("--repetition-penalty", type=float, default=None)
    generate.add_argument("--eos-token-id", type=int, default=None)
    generate.add_argument("--pad-token-id", type=int, default=None)
    generate.add_argument("--local-window-size", type=int, default=None)
    generate.add_argument("--max-cache-length", type=int, default=None)
    generate.add_argument("--use-mtp-draft", action="store_true")
    generate.add_argument("--mtp-accept-mode", choices=["greedy", "match_main"], default="greedy")
    generate.add_argument("--max-mtp-draft-tokens", type=int, default=None)
    generate.add_argument("--return-cache-stats", action="store_true")
    generate.add_argument("--include-scores", action="store_true")
    generate.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
