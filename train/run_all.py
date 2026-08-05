#!/usr/bin/env python3
"""Train all pairs (30 epochs) then generate full plot reports."""

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
    parser.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=list(PAIRS))
    parser.add_argument(
        "--python",
        default=str(TRAIN_DIR / ".venv" / "Scripts" / "python.exe"),
    )
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--no-cv", action="store_true", help="Skip k-fold CV in report")
    args = parser.parse_args()

    py = Path(args.python)
    if not py.is_file():
        raise SystemExit(f"Python not found: {py}")

    log_dir = TRAIN_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[str] = []
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    for pair in args.pairs:
        cfg = TRAIN_DIR / "configs" / f"{pair}.yaml"
        log_file = log_dir / f"train30_{pair}.log"
        print(f"\n======== TRAIN 30 EPOCHS {pair} ========", flush=True)
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
        line = f"{pair} train: {status} in {elapsed/3600:.2f}h -> {log_file}"
        print(f"======== DONE {line} ========", flush=True)
        results.append(line)
        if proc.returncode != 0:
            continue

        if not args.skip_report:
            report_log = log_dir / f"report_{pair}.log"
            print(f"======== REPORT {pair} ========", flush=True)
            cmd = [
                str(py),
                str(TRAIN_DIR / "generate_reports.py"),
                "--pairs",
                pair,
                "--eval-samples",
                "500",
                "--cv-samples",
                "4000",
                "--cv-folds",
                "3",
                "--cv-epochs",
                "2",
            ]
            if args.no_cv:
                cmd.append("--no-cv")
            with report_log.open("w", encoding="utf-8") as log:
                r2 = subprocess.run(
                    cmd, cwd=str(TRAIN_DIR), stdout=log, stderr=subprocess.STDOUT, env=env
                )
            results.append(
                f"{pair} report: {'OK' if r2.returncode == 0 else f'FAIL({r2.returncode})'} -> {report_log}"
            )

    summary = log_dir / "train30_all_summary.txt"
    summary.write_text("\n".join(results) + "\n", encoding="utf-8")
    print("\nSummary:", summary, flush=True)
    raise SystemExit(1 if any("FAIL" in r for r in results) else 0)


if __name__ == "__main__":
    main()
