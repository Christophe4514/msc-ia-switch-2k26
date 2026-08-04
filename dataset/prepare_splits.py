#!/usr/bin/env python3
"""Clean OPUS moses bitexts and split train/valid/test (80/10/10).

Phase 1 of the thesis pipeline:
  - merge corpora per language pair
  - normalize / filter noisy pairs
  - optional LASER score filter for NLLB
  - deduplicate
  - stratified-free random split with fixed seed
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Folder name -> target file extensions accepted as the "target" side
PAIR_TARGETS: dict[str, tuple[str, ...]] = {
    "fr-ln": ("ln",),
    "fr-kg": ("kg",),
    "fr-lu": ("lua", "lu"),
    "fr-sw": ("sw", "swc"),
}

# Canonical target code written in output filenames
PAIR_CANONICAL_TGT: dict[str, str] = {
    "fr-ln": "ln",
    "fr-kg": "kg",
    "fr-lu": "lua",
    "fr-sw": "sw",
}

WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")


@dataclass
class CleanConfig:
    min_chars: int = 3
    max_chars: int = 500
    max_words: int = 100
    max_len_ratio: float = 3.0
    min_laser_score: float | None = 1.05  # None = keep all scored pairs
    train_ratio: float = 0.80
    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42


@dataclass
class FilterStats:
    raw_pairs: int = 0
    kept: int = 0
    empty: int = 0
    too_short: int = 0
    too_long: int = 0
    bad_ratio: int = 0
    identical: int = 0
    low_score: int = 0
    duplicates: int = 0
    sources: Counter = field(default_factory=Counter)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def discover_bitexts(pair_dir: Path, tgt_exts: tuple[str, ...]) -> list[tuple[Path, Path, Path | None, str]]:
    """Return list of (src.fr, tgt, scores|None, corpus_name)."""
    moses = pair_dir / "moses"
    if not moses.is_dir():
        return []

    found: list[tuple[Path, Path, Path | None, str]] = []
    for corpus_dir in sorted(moses.iterdir()):
        if not corpus_dir.is_dir():
            continue
        fr_files = list(corpus_dir.glob("*.fr"))
        if not fr_files:
            continue
        src = fr_files[0]
        tgt = None
        for ext in tgt_exts:
            candidates = list(corpus_dir.glob(f"*.{ext}"))
            if candidates:
                tgt = candidates[0]
                break
        if tgt is None:
            continue
        scores = corpus_dir / (src.name[: -len(".fr")] + ".scores")
        if not scores.is_file():
            # NLLB.fr-ln.scores style: stem without language
            alt = list(corpus_dir.glob("*.scores"))
            scores_path = alt[0] if alt else None
        else:
            scores_path = scores
        found.append((src, tgt, scores_path, corpus_dir.name))
    return found


def iter_aligned(
    src_path: Path, tgt_path: Path, scores_path: Path | None
):
    with src_path.open("r", encoding="utf-8", errors="replace") as fs, tgt_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as ft:
        if scores_path and scores_path.is_file():
            with scores_path.open("r", encoding="utf-8", errors="replace") as fsc:
                for s, t, sc in zip(fs, ft, fsc):
                    try:
                        score = float(sc.strip())
                    except ValueError:
                        score = None
                    yield s, t, score
        else:
            for s, t in zip(fs, ft):
                yield s, t, None


def clean_pair(
    bitexts: list[tuple[Path, Path, Path | None, str]],
    cfg: CleanConfig,
) -> tuple[list[tuple[str, str]], FilterStats]:
    stats = FilterStats()
    # key -> (src, tgt, score)  keep best score on duplicate
    best: dict[tuple[str, str], tuple[str, str, float]] = {}

    for src_path, tgt_path, scores_path, corpus_name in bitexts:
        for raw_s, raw_t, score in iter_aligned(src_path, tgt_path, scores_path):
            stats.raw_pairs += 1
            stats.sources[corpus_name] += 1

            s = normalize_text(raw_s)
            t = normalize_text(raw_t)

            if not s or not t:
                stats.empty += 1
                continue
            if len(s) < cfg.min_chars or len(t) < cfg.min_chars:
                stats.too_short += 1
                continue
            if (
                len(s) > cfg.max_chars
                or len(t) > cfg.max_chars
                or word_count(s) > cfg.max_words
                or word_count(t) > cfg.max_words
            ):
                stats.too_long += 1
                continue

            ls, lt = len(s), len(t)
            ratio = max(ls, lt) / max(1, min(ls, lt))
            if ratio > cfg.max_len_ratio:
                stats.bad_ratio += 1
                continue

            if s.casefold() == t.casefold():
                stats.identical += 1
                continue

            if (
                cfg.min_laser_score is not None
                and score is not None
                and score < cfg.min_laser_score
            ):
                stats.low_score += 1
                continue

            key = (s.casefold(), t.casefold())
            score_val = score if score is not None else 0.0
            if key in best:
                stats.duplicates += 1
                if score_val <= best[key][2]:
                    continue
            best[key] = (s, t, score_val)

    pairs = [(s, t) for s, t, _ in best.values()]
    stats.kept = len(pairs)
    return pairs, stats


def split_pairs(
    pairs: list[tuple[str, str]], cfg: CleanConfig
) -> dict[str, list[tuple[str, str]]]:
    assert abs(cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio - 1.0) < 1e-6
    rng = random.Random(cfg.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * cfg.train_ratio)
    n_valid = int(n * cfg.valid_ratio)
    # remainder -> test (handles rounding)
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train : n_train + n_valid],
        "test": shuffled[n_train + n_valid :],
    }


def write_split(
    out_dir: Path,
    splits: dict[str, list[tuple[str, str]]],
    tgt_code: str,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, rows in splits.items():
        src_path = out_dir / f"{name}.fr"
        tgt_path = out_dir / f"{name}.{tgt_code}"
        with src_path.open("w", encoding="utf-8", newline="\n") as fs, tgt_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as ft:
            for s, t in rows:
                fs.write(s + "\n")
                ft.write(t + "\n")
        counts[name] = len(rows)
    return counts


def process_pair(pair: str, cfg: CleanConfig) -> dict:
    pair_dir = ROOT / pair
    tgt_exts = PAIR_TARGETS[pair]
    tgt_code = PAIR_CANONICAL_TGT[pair]
    bitexts = discover_bitexts(pair_dir, tgt_exts)
    if not bitexts:
        raise FileNotFoundError(f"No moses bitexts found for {pair}")

    pairs, stats = clean_pair(bitexts, cfg)
    splits = split_pairs(pairs, cfg)
    out_dir = pair_dir / "splits"
    counts = write_split(out_dir, splits, tgt_code)

    report = {
        "pair": pair,
        "target": tgt_code,
        "config": asdict(cfg),
        "corpora": [b[3] for b in bitexts],
        "filter": {
            **{k: v for k, v in asdict(stats).items() if k != "sources"},
            "sources": dict(stats.sources),
        },
        "split_counts": counts,
        "output_dir": str(out_dir.relative_to(ROOT)),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=list(PAIR_TARGETS),
        choices=list(PAIR_TARGETS),
        help="Language pairs to process (default: all)",
    )
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--max-words", type=int, default=100)
    parser.add_argument("--max-len-ratio", type=float, default=3.0)
    parser.add_argument(
        "--min-laser-score",
        type=float,
        default=1.05,
        help="Drop NLLB pairs below this LASER score (use -1 to disable)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    min_score = None if args.min_laser_score < 0 else args.min_laser_score
    cfg = CleanConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_words=args.max_words,
        max_len_ratio=args.max_len_ratio,
        min_laser_score=min_score,
        seed=args.seed,
    )

    summary = []
    for pair in args.pairs:
        print(f"=== {pair} ===")
        report = process_pair(pair, cfg)
        f = report["filter"]
        c = report["split_counts"]
        print(
            f"  raw={f['raw_pairs']:,}  kept={f['kept']:,}  "
            f"train={c['train']:,}  valid={c['valid']:,}  test={c['test']:,}"
        )
        print(
            f"  filtered: empty={f['empty']} short={f['too_short']} "
            f"long={f['too_long']} ratio={f['bad_ratio']} "
            f"identical={f['identical']} low_score={f['low_score']} "
            f"dup={f['duplicates']}"
        )
        summary.append(report)

    out = ROOT / "splits_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
