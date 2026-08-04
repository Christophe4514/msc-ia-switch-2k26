#!/usr/bin/env python3
"""Interactive / CLI testing of a trained LoRA translation model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_config, load_model, resolve_path, translate_batch  # noqa: E402


def run_test(
    cfg: dict,
    *,
    text: str | None = None,
    file: Path | None = None,
    interactive: bool = False,
    adapter: Path | None = None,
    baseline: bool = False,
) -> None:
    output_dir = resolve_path(cfg["output_dir"])
    adapter_dir = adapter or (output_dir / "adapter")
    model, tokenizer, device = load_model(cfg, adapter_dir, baseline)
    max_len = int(cfg.get("generation_max_length", 128))

    def translate_one(src: str) -> str:
        outs = translate_batch(
            model,
            tokenizer,
            [src],
            src_lang=cfg["src_lang"],
            tgt_lang=cfg["tgt_lang"],
            max_length=max_len,
            device=device,
        )
        return outs[0]

    print(f"Model : {cfg['model_name']}")
    print(f"Pair  : {cfg['pair']} ({cfg['src_lang']} -> {cfg['tgt_lang']})")
    print(f"Mode  : {'baseline' if baseline else f'LoRA ({adapter_dir})'}")
    print("-" * 50)

    if text:
        print("FR :", text)
        print(">> :", translate_one(text))
        return

    if file:
        lines = Path(file).read_text(encoding="utf-8").splitlines()
        out_path = output_dir / f"test_out_{cfg['pair']}.txt"
        with out_path.open("w", encoding="utf-8") as fout:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                hyp = translate_one(line)
                print(f"FR : {line}")
                print(f">> : {hyp}\n")
                fout.write(f"{line}\t{hyp}\n")
        print(f"Wrote {out_path}")
        return

    # interactive by default
    print("Mode interactif — tapez une phrase FR (quit / exit pour quitter).")
    while True:
        try:
            src = input("FR > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not src:
            continue
        if src.lower() in {"quit", "exit", "q"}:
            break
        print(">> :", translate_one(src))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--text", type=str, default=None, help="Single French sentence")
    parser.add_argument("--file", type=Path, default=None, help="File with one FR sentence per line")
    parser.add_argument("--interactive", action="store_true", help="REPL mode")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
