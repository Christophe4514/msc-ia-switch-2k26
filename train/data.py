"""Load Moses-style train/valid/test splits into Hugging Face Datasets."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict


def load_moses_pair(
    splits_dir: Path,
    tgt_ext: str,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    splits_dir = Path(splits_dir)
    data = {}
    for split, limit in (
        ("train", max_train_samples),
        ("valid", max_eval_samples),
        ("test", max_eval_samples),
    ):
        src_path = splits_dir / f"{split}.fr"
        tgt_path = splits_dir / f"{split}.{tgt_ext}"
        if not src_path.is_file() or not tgt_path.is_file():
            raise FileNotFoundError(
                f"Missing split files for '{split}' in {splits_dir} "
                f"(expected {src_path.name} and {tgt_path.name}). "
                "Run: python dataset/prepare_splits.py"
            )
        with src_path.open(encoding="utf-8") as fs, tgt_path.open(encoding="utf-8") as ft:
            sources = [line.rstrip("\n") for line in fs]
            targets = [line.rstrip("\n") for line in ft]
        if len(sources) != len(targets):
            raise ValueError(
                f"Line count mismatch in {split}: {len(sources)} src vs {len(targets)} tgt"
            )
        if limit is not None:
            sources = sources[:limit]
            targets = targets[:limit]
        data[split] = Dataset.from_dict({"source": sources, "target": targets})
    return DatasetDict(data)


def preprocess_function(
    examples,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    max_source_length: int,
    max_target_length: int,
):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=examples["target"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
