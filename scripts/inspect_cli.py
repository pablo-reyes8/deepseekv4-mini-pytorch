from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import torch

from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig


TEST_TARGETS = {
    "rmsnorm": ["tests/test_RMSNorm.py"],
    "rope": ["tests/test_rope.py"],
    "embedding": ["tests/test_embedding.py"],
    "mha": ["tests/test_mha_baseline.py"],
    "hca": ["tests/test_hca.py"],
    "csa": ["tests/test_csa.py"],
    "moe": ["tests/test_moe.py"],
    "mhc": ["tests/test_mhc.py"],
    "mtp": ["tests/test_mtp.py"],
    "block": ["tests/test_transformer_block.py", "tests/test_deepseek_model.py"],
    "training": ["tests/training"],
    "data": ["tests/data"],
    "parallel": ["tests/parallel"],
    "inference": ["tests/inference"],
    "all": ["tests"],
}


def print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def make_model(args: argparse.Namespace) -> DeepSeekV4LM:
    return DeepSeekV4LM(
        DeepSeekV4LMConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            pad_token_id=0,
            attention_type=args.attention,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            rotary_dim=args.rotary_dim or args.head_dim,
            compression_factor=args.compression_factor,
            hca_compression_factor=args.compression_factor,
            top_k_blocks=args.top_k_blocks,
            window_size=args.window_size,
            indexer_dim=args.indexer_dim,
            n_indexer_heads=args.n_indexer_heads,
            query_compression_dim=args.indexer_dim,
            ffn_type=args.ffn,
            mlp_hidden_dim=args.mlp_hidden_dim,
            num_experts=args.num_experts,
            top_k_experts=args.top_k_experts,
            expert_hidden_dim=args.expert_hidden_dim,
            shared_hidden_dim=args.expert_hidden_dim,
            use_mhc=args.use_mhc,
            n_hc=args.n_hc,
            mhc_sinkhorn_iters=args.mhc_sinkhorn_iters,
            use_mtp=args.use_mtp,
            mtp_depth=args.mtp_depth,
            mtp_hidden_dim=args.d_model,
        )
    )


def cmd_model_summary(args: argparse.Namespace) -> None:
    model = make_model(args)
    by_top_module = defaultdict(int)

    for name, param in model.named_parameters():
        by_top_module[name.split(".", 1)[0]] += param.numel()

    input_ids = torch.randint(1, args.vocab_size, (args.batch_size, args.max_seq_len))
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    print_json(
        {
            "class": type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameters_by_top_module": dict(sorted(by_top_module.items())),
            "forward": {
                "input_ids_shape": list(input_ids.shape),
                "logits_shape": list(outputs["logits"].shape),
                "logits_finite": bool(torch.isfinite(outputs["logits"]).all().item()),
            },
            "config": {
                "attention": args.attention,
                "ffn": args.ffn,
                "use_mhc": args.use_mhc,
                "use_mtp": args.use_mtp,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
            },
        }
    )


def cmd_module_tests(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "pytest", *TEST_TARGETS[args.module]]
    if args.quiet:
        cmd.append("-q")
    env = dict(os.environ)
    env.setdefault("TMPDIR", "/tmp")
    raise SystemExit(subprocess.call(cmd, env=env))


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attention", choices=["mha", "hca", "csa"], default="mha")
    parser.add_argument("--ffn", choices=["dense", "moe"], default="dense")
    parser.add_argument("--use-mhc", action="store_true")
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--rotary-dim", type=int, default=None)
    parser.add_argument("--compression-factor", type=int, default=4)
    parser.add_argument("--top-k-blocks", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--indexer-dim", type=int, default=8)
    parser.add_argument("--n-indexer-heads", type=int, default=2)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k-experts", type=int, default=2)
    parser.add_argument("--expert-hidden-dim", type=int, default=64)
    parser.add_argument("--n-hc", type=int, default=2)
    parser.add_argument("--mhc-sinkhorn-iters", type=int, default=5)
    parser.add_argument("--mtp-depth", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Mini inspection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_summary = subparsers.add_parser("model-summary", help="Build and inspect a model")
    add_model_args(model_summary)
    model_summary.set_defaults(func=cmd_model_summary)

    module_tests = subparsers.add_parser("module-tests", help="Run tests for one module group")
    module_tests.add_argument("module", choices=sorted(TEST_TARGETS))
    module_tests.add_argument("--quiet", action="store_true")
    module_tests.set_defaults(func=cmd_module_tests)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
