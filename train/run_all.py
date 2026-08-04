#!/usr/bin/env python3
"""Train LoRA adapters for all FR→national language pairs sequentially."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
PAIRS = ("fr-ln", "fr-kg", "fr-lu", "fr-sw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=list(PAIRS),
        choices=list(PAIRS),
        help="Subset of pairs (default: all)",
    )
    parser.add_argument(
        "--python",
        default=str(TRAIN_DIR / ".venv" / "Scripts" / "python.exe"),
        help="Python executable (default: train/.venv)",
    )
    args = parser.parse_args()

    py = Path(args.python)
    if not py.is_file():
        raise SystemExit(f"Python not found: {py}. Create the venv first (see README).")

    log_dir = TRAIN_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "run_all_summary.txt"
    results: list[str] = []

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    for pair in args.pairs:
        cfg = TRAIN_DIR / "configs" / f"{pair}.yaml"
        log_file = log_dir / f"train_{pair}.log"
        print(f"\n======== START {pair} ========", flush=True)
        print(f"config={cfg}", flush=True)
        print(f"log={log_file}", flush=True)
        t0 = time.time()
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [str(py), str(TRAIN_DIR / "train_lora.py"), "--config", str(cfg)],
                cwd=str(TRAIN_DIR),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        line = f"{pair}: {status} in {elapsed/3600:.2f}h -> {log_file}"
        print(f"======== DONE {line} ========", flush=True)
        results.append(line)
        if proc.returncode != 0:
            results.append(f"  See log: {log_file}")

    summary_path.write_text("\n".join(results) + "\n", encoding="utf-8")
    print("\nSummary written to", summary_path, flush=True)
    failed = [r for r in results if "FAIL" in r]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
