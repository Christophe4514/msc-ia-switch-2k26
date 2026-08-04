"""Save training / evaluation charts as PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )


def plot_training_curves(log_history: list[dict], out_dir: Path, pair: str) -> list[Path]:
    """Plot loss / BLEU / WER / accuracy from Trainer log_history."""
    _style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    train_loss = [
        (h.get("epoch"), h["loss"])
        for h in log_history
        if "loss" in h and "eval_loss" not in h and h.get("epoch") is not None
    ]
    eval_loss = [
        (h.get("epoch"), h["eval_loss"])
        for h in log_history
        if "eval_loss" in h and h.get("epoch") is not None
    ]
    eval_bleu = [
        (h.get("epoch"), h["eval_bleu"])
        for h in log_history
        if "eval_bleu" in h and h.get("epoch") is not None
    ]
    eval_wer = [
        (h.get("epoch"), h["eval_wer"])
        for h in log_history
        if "eval_wer" in h and h.get("epoch") is not None
    ]
    eval_acc = [
        (h.get("epoch"), h["eval_accuracy"])
        for h in log_history
        if "eval_accuracy" in h and h.get("epoch") is not None
    ]
    eval_chrf = [
        (h.get("epoch"), h["eval_chrf"])
        for h in log_history
        if "eval_chrf" in h and h.get("epoch") is not None
    ]

    # 1) Loss
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if train_loss:
        ax.plot([e for e, _ in train_loss], [v for _, v in train_loss], label="train loss")
    if eval_loss:
        ax.plot(
            [e for e, _ in eval_loss],
            [v for _, v in eval_loss],
            marker="o",
            label="eval loss",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{pair} — Loss")
    ax.legend()
    path = out_dir / f"{pair}_loss.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 2) BLEU + chrF
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if eval_bleu:
        ax.plot(
            [e for e, _ in eval_bleu],
            [v for _, v in eval_bleu],
            marker="o",
            label="BLEU",
        )
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, label="Qmin=25")
    if eval_chrf:
        ax.plot(
            [e for e, _ in eval_chrf],
            [v for _, v in eval_chrf],
            marker="s",
            label="chrF",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"{pair} — BLEU / chrF")
    ax.legend()
    path = out_dir / f"{pair}_bleu_chrf.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 3) WER + accuracy
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if eval_wer:
        ax.plot(
            [e for e, _ in eval_wer],
            [v for _, v in eval_wer],
            marker="o",
            color="tab:red",
            label="WER (%)",
        )
    if eval_acc:
        ax.plot(
            [e for e, _ in eval_acc],
            [v for _, v in eval_acc],
            marker="s",
            color="tab:green",
            label="Accuracy (%)",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percent")
    ax.set_title(f"{pair} — WER / Accuracy")
    ax.legend()
    path = out_dir / f"{pair}_wer_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 4) Combined dashboard
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{pair} — Training dashboard", fontsize=14)

    ax = axes[0, 0]
    if train_loss:
        ax.plot([e for e, _ in train_loss], [v for _, v in train_loss], label="train")
    if eval_loss:
        ax.plot([e for e, _ in eval_loss], [v for _, v in eval_loss], marker="o", label="eval")
    ax.set_title("Loss")
    ax.legend()

    ax = axes[0, 1]
    if eval_bleu:
        ax.plot([e for e, _ in eval_bleu], [v for _, v in eval_bleu], marker="o", label="BLEU")
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=1)
    if eval_chrf:
        ax.plot([e for e, _ in eval_chrf], [v for _, v in eval_chrf], marker="s", label="chrF")
    ax.set_title("BLEU / chrF")
    ax.legend()

    ax = axes[1, 0]
    if eval_wer:
        ax.plot([e for e, _ in eval_wer], [v for _, v in eval_wer], marker="o", color="tab:red")
    ax.set_title("WER (%)")

    ax = axes[1, 1]
    if eval_acc:
        ax.plot(
            [e for e, _ in eval_acc],
            [v for _, v in eval_acc],
            marker="s",
            color="tab:green",
        )
    ax.set_title("Accuracy (%)")

    for ax in axes.ravel():
        ax.set_xlabel("Epoch")

    path = out_dir / f"{pair}_dashboard.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    return written


def plot_final_metrics_bar(metrics: dict, out_path: Path, title: str) -> Path:
    """Bar chart for a single evaluation (BLEU, chrF, WER, accuracy)."""
    _style()
    keys = ["bleu", "chrf", "wer", "accuracy"]
    labels = ["BLEU", "chrF", "WER", "Accuracy"]
    values = [float(metrics.get(k, 0.0)) for k in keys]
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#16a34a"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Score")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    if "bleu" in metrics:
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, label="Qmin BLEU=25")
        ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
