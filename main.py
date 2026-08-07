#!/usr/bin/env python3
"""
Point d'entrée général — msc-ia-switch-2k26

Usage:
  python main.py train --config train/configs/fr-ln.yaml
  python main.py train-all
  python main.py evaluate --config train/configs/fr-ln.yaml --split test
  python main.py test --config train/configs/fr-ln.yaml --interactive
  python main.py report --pairs fr-ln fr-kg fr-lu fr-sw
  python main.py export --config train/configs/fr-ln.yaml --baseline
  python main.py export --config train/configs/fr-ln.yaml --lora --int8
  python main.py serve --baseline
  python main.py serve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train"
sys.path.insert(0, str(TRAIN))


def cmd_train(args: argparse.Namespace) -> None:
    from train_lora import load_config, run_train

    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        cfg["max_eval_samples"] = args.max_eval_samples
    run_train(cfg)


def cmd_train_all(args: argparse.Namespace) -> None:
    from run_all import main as run_all_main

    argv = []
    if args.pairs:
        argv.extend(["--pairs", *args.pairs])
    sys.argv = [sys.argv[0], *argv]
    run_all_main()


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate import load_config, run_evaluate

    cfg = load_config(args.config)
    run_evaluate(
        cfg,
        split=args.split,
        adapter=args.adapter,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        baseline=args.baseline,
    )


def cmd_test(args: argparse.Namespace) -> None:
    from evaluate import load_config
    from test_model import run_test

    cfg = load_config(args.config)
    interactive = args.interactive or (args.text is None and args.file is None)
    run_test(
        cfg,
        text=args.text,
        file=args.file,
        interactive=interactive,
        adapter=args.adapter,
        baseline=args.baseline,
    )


def cmd_report(args: argparse.Namespace) -> None:
    from generate_reports import generate_report, load_config

    for pair in args.pairs:
        cfg = load_config(TRAIN / "configs" / f"{pair}.yaml")
        generate_report(
            cfg,
            with_cv=not args.no_cv,
            cv_samples=args.cv_samples,
            cv_epochs=args.cv_epochs,
            cv_folds=args.cv_folds,
            eval_samples=args.eval_samples,
        )


def cmd_export(args: argparse.Namespace) -> None:
    from export_onnx import export_pair, load_config

    cfg = load_config(args.config)
    export_pair(
        cfg,
        baseline=not args.lora,
        adapter=args.adapter,
        quantize_int8=args.int8,
    )


def cmd_serve(args: argparse.Namespace) -> None:
    from serve_api import run_server

    run_server(
        args.pairs,
        host=args.host,
        port=args.port,
        baseline=args.baseline,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="MSC IA Switch — train / evaluate / test / report / export NLLB-LoRA",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fine-tune LoRA + export PNG metrics")
    p_train.add_argument(
        "--config",
        type=Path,
        default=TRAIN / "configs" / "fr-ln.yaml",
    )
    p_train.add_argument("--max-train-samples", type=int, default=None)
    p_train.add_argument("--max-eval-samples", type=int, default=None)
    p_train.set_defaults(func=cmd_train)

    p_all = sub.add_parser("train-all", help="Train all language pairs sequentially")
    p_all.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        choices=["fr-ln", "fr-kg", "fr-lu", "fr-sw"],
    )
    p_all.set_defaults(func=cmd_train_all)

    p_eval = sub.add_parser("evaluate", help="BLEU/chrF/WER/accuracy + PNG bar chart")
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--adapter", type=Path, default=None)
    p_eval.add_argument("--split", choices=["valid", "test"], default="test")
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--max-samples", type=int, default=None)
    p_eval.add_argument("--baseline", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_test = sub.add_parser("test", help="Translate FR text with the trained model")
    p_test.add_argument("--config", type=Path, required=True)
    p_test.add_argument("--adapter", type=Path, default=None)
    p_test.add_argument("--text", type=str, default=None)
    p_test.add_argument("--file", type=Path, default=None)
    p_test.add_argument("--interactive", action="store_true")
    p_test.add_argument("--baseline", action="store_true")
    p_test.set_defaults(func=cmd_test)

    p_report = sub.add_parser(
        "report",
        help="Generate all thesis plots (accuracy/loss/CM/CV/scores + architecture)",
    )
    p_report.add_argument(
        "--pairs",
        nargs="*",
        default=["fr-ln", "fr-kg", "fr-lu", "fr-sw"],
        choices=["fr-ln", "fr-kg", "fr-lu", "fr-sw"],
    )
    p_report.add_argument("--no-cv", action="store_true")
    p_report.add_argument("--cv-samples", type=int, default=6000)
    p_report.add_argument("--cv-epochs", type=int, default=2)
    p_report.add_argument("--cv-folds", type=int, default=5)
    p_report.add_argument("--eval-samples", type=int, default=500)
    p_report.set_defaults(func=cmd_report)

    p_export = sub.add_parser(
        "export",
        help="Export baseline or LoRA-merged model to ONNX for Flutter",
    )
    p_export.add_argument("--config", type=Path, required=True)
    p_export.add_argument(
        "--baseline",
        action="store_true",
        help="Export base NLLB (default if --lora not set)",
    )
    p_export.add_argument(
        "--lora",
        action="store_true",
        help="Merge LoRA adapter then export",
    )
    p_export.add_argument("--adapter", type=Path, default=None)
    p_export.add_argument(
        "--int8",
        action="store_true",
        help="Dynamic INT8 quantization (Optimum)",
    )
    p_export.set_defaults(func=cmd_export)

    p_serve = sub.add_parser(
        "serve",
        help="Expose le modèle (LoRA/baseline) en HTTP pour l'app Flutter",
    )
    p_serve.add_argument(
        "--pairs",
        nargs="*",
        default=["fr-ln", "fr-kg", "fr-lu", "fr-sw"],
        choices=["fr-ln", "fr-kg", "fr-lu", "fr-sw"],
    )
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument(
        "--baseline",
        action="store_true",
        help="Utiliser NLLB de base si LoRA pas encore entraîné",
    )
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
