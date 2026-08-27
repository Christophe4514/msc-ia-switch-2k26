#!/usr/bin/env python3
"""Regénère loss / accuracy / scores depuis trainer_history.json uniquement.

Usage:
  python train/regenerate_plots.py
  python train/regenerate_plots.py --pairs fr-sw fr-ln
  python train/regenerate_plots.py --max-train-points 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plotting import plot_accuracy, plot_loss, plot_scores_bar  # noqa: E402

PAIRS = ("fr-ln", "fr-kg", "fr-lu", "fr-sw")
MAX_EPOCHS = 12


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Historique introuvable: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Format attendu: liste JSON — {path}")
    return data


def _epoch_key(epoch: float) -> int:
    """Mappe un epoch HF (float) vers 1..MAX_EPOCHS."""
    ep = int(round(float(epoch)))
    return max(1, min(MAX_EPOCHS, ep))


def extract_curves(history: list[dict]) -> dict:
    """Extrait séries train / val depuis log_history."""
    train_epochs: list[float] = []
    train_loss: list[float] = []

    val_loss_by_ep: dict[int, float] = {}
    val_acc_by_ep: dict[int, float] = {}
    scores: dict[str, float] = {}

    for row in history:
        epoch = row.get("epoch")
        if epoch is None:
            continue

        # Train loss (logging_steps — beaucoup de points)
        if "loss" in row and "eval_loss" not in row and "train_loss" not in row:
            train_epochs.append(float(epoch))
            train_loss.append(float(row["loss"]))

        # Validation loss — une valeur par epoch (dernière gagne)
        if "eval_loss" in row:
            val_loss_by_ep[_epoch_key(epoch)] = float(row["eval_loss"])

        # Accuracy % — ignore les placeholders ~0.5 (eval sans generate)
        if "eval_accuracy" in row:
            acc = float(row["eval_accuracy"])
            if acc >= 1.0:
                val_acc_by_ep[_epoch_key(epoch)] = acc

        # Scores finaux (BLEU / chrF / WER / accuracy) — garder le dernier
        if "eval_bleu" in row:
            scores = {
                "bleu": float(row.get("eval_bleu", 0)),
                "chrf": float(row.get("eval_chrf", 0)),
                "wer": float(row.get("eval_wer", 0)),
                "accuracy": float(row.get("eval_accuracy", 0)),
            }

    epoch_axis = list(range(1, MAX_EPOCHS + 1))
    val_loss = [val_loss_by_ep.get(e, np.nan) for e in epoch_axis]
    val_acc = [val_acc_by_ep.get(e, np.nan) for e in epoch_axis]

    return {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "epoch_axis": epoch_axis,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "scores": scores,
    }


def subsample_train(
    epochs: list[float],
    losses: list[float],
    max_points: int,
) -> tuple[list[float], list[float]]:
    if len(epochs) <= max_points:
        return epochs, losses
    idx = np.linspace(0, len(epochs) - 1, max_points, dtype=int)
    return [epochs[i] for i in idx], [losses[i] for i in idx]


def _drop_nan_pairs(xs: list, ys: list) -> tuple[list, list]:
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y


def regenerate_pair(
    pair: str,
    *,
    outputs_root: Path,
    max_train_points: int = 50,
) -> list[Path]:
    output_dir = outputs_root / f"nllb-lora-{pair}"
    history_path = output_dir / "trainer_history.json"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    curves = extract_curves(load_history(history_path))
    written: list[Path] = []

    # --- Loss (train subsample + val 12 epochs) ---
    tx, ty = subsample_train(
        curves["train_epochs"],
        curves["train_loss"],
        max_train_points,
    )
    vx, vy = _drop_nan_pairs(curves["epoch_axis"], curves["val_loss"])
    written.append(
        plot_loss(
            tx,
            ty,
            vy,
            plots_dir / f"{pair}_2_loss.png",
            pair,
            val_x=vx,
            train_x=tx,
        )
    )

    # --- Accuracy (12 epochs, points réels uniquement) ---
    ax, ay = _drop_nan_pairs(curves["epoch_axis"], curves["val_acc"])
    written.append(
        plot_accuracy(
            ax or curves["epoch_axis"],
            [],
            ay,
            plots_dir / f"{pair}_1_accuracy.png",
            pair,
        )
    )

    # --- Scores bar (dernier eval avec BLEU) ---
    if curves["scores"]:
        written.append(
            plot_scores_bar(
                curves["scores"],
                plots_dir / f"{pair}_6_scores.png",
                pair,
            )
        )
    else:
        print(f"  [{pair}] pas de eval_bleu dans l'historique — scores ignorés")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=list(PAIRS),
        choices=list(PAIRS),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT / "outputs",
    )
    parser.add_argument(
        "--max-train-points",
        type=int,
        default=50,
        help="Points max pour la courbe train loss (lisibilité)",
    )
    args = parser.parse_args()

    for pair in args.pairs:
        print(f"=== {pair} ===")
        try:
            paths = regenerate_pair(
                pair,
                outputs_root=args.outputs_root,
                max_train_points=args.max_train_points,
            )
            for p in paths:
                print(f"  -> {p}")
        except FileNotFoundError as exc:
            print(f"  skip: {exc}")


if __name__ == "__main__":
    main()
