"""Translation quality metrics: BLEU, chrF, WER, sentence accuracy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sacrebleu import CHRF, corpus_bleu


@dataclass
class TranslationMetrics:
    bleu: float
    chrf: float
    wer: float
    accuracy: float  # exact sentence match rate (%)
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def _tokenize_words(text: str) -> list[str]:
    return text.strip().split()


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance on word tokens."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def word_error_rate(hypotheses: list[str], references: list[str]) -> float:
    """Corpus WER in percent (0 = perfect)."""
    total_dist = 0
    total_ref = 0
    for hyp, ref in zip(hypotheses, references):
        r = _tokenize_words(ref)
        h = _tokenize_words(hyp)
        total_dist += _edit_distance(r, h)
        total_ref += max(len(r), 1)
    return 100.0 * total_dist / total_ref


def sentence_accuracy(hypotheses: list[str], references: list[str]) -> float:
    """Exact-match sentence accuracy in percent."""
    if not hypotheses:
        return 0.0
    hits = sum(
        1
        for h, r in zip(hypotheses, references)
        if h.strip().casefold() == r.strip().casefold()
    )
    return 100.0 * hits / len(hypotheses)


def compute_translation_metrics(
    hypotheses: list[str], references: list[str]
) -> TranslationMetrics:
    bleu = corpus_bleu(hypotheses, [references]).score
    chrf = CHRF().corpus_score(hypotheses, [references]).score
    return TranslationMetrics(
        bleu=float(bleu),
        chrf=float(chrf),
        wer=float(word_error_rate(hypotheses, references)),
        accuracy=float(sentence_accuracy(hypotheses, references)),
        n=len(hypotheses),
    )


def decode_preds_labels(tokenizer, preds, labels):
    """Decode trainer predictions/labels, replacing -100."""
    import numpy as np

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    hyp = tokenizer.batch_decode(preds, skip_special_tokens=True)
    ref = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return hyp, ref
