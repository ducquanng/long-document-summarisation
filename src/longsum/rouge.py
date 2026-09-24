"""ROUGE, written out rather than imported.

ROUGE compares a generated summary with a reference by counting overlap: unigrams
(ROUGE-1), bigrams (ROUGE-2) and the longest common subsequence (ROUGE-L). It is
a crude measure - it rewards copying the reference's wording and cannot tell a
true statement from a false one - but it is the measure the summarisation
literature reports, so it is the one to report alongside its caveats.

Implemented here because a dependency would hide the definition, and the point of
this repository is that the definition matters.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Score:
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate: str, reference: str, n: int = 1) -> Score:
    """Overlapping n-grams, counted with multiplicity capped at the reference's."""
    cand, ref = _ngrams(tokenize(candidate), n), _ngrams(tokenize(reference), n)
    overlap = sum((cand & ref).values())
    cand_total, ref_total = sum(cand.values()), sum(ref.values())
    return Score(
        precision=overlap / cand_total if cand_total else 0.0,
        recall=overlap / ref_total if ref_total else 0.0,
    )


def lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence, the basis of ROUGE-L.

    Unlike n-gram overlap this rewards words appearing in the right *order* even
    when other words come between them, which is why it is the variant that
    notices when a summary has been assembled out of sequence.
    """
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            current.append(previous[j] + 1 if token_a == token_b else max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> Score:
    cand, ref = tokenize(candidate), tokenize(reference)
    length = lcs_length(cand, ref)
    return Score(
        precision=length / len(cand) if cand else 0.0,
        recall=length / len(ref) if ref else 0.0,
    )


def all_scores(candidate: str, reference: str) -> dict[str, float]:
    return {
        "rouge1": rouge_n(candidate, reference, 1).f1,
        "rouge2": rouge_n(candidate, reference, 2).f1,
        "rougeL": rouge_l(candidate, reference).f1,
    }
