"""A sample of long news articles with reference summaries.

CNN/DailyMail: each article ships with the bullet-point highlights a human wrote,
which serve as the reference. Only articles that overflow the model's window are
kept, because a document that fits needs none of this.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable

from .config import DATA_DIR, MIN_DOCUMENT_TOKENS, N_DOCUMENTS

CACHE = DATA_DIR / "sample.jsonl"


def build_sample(count_tokens: Callable[[str], int], n: int = N_DOCUMENTS,
                 min_tokens: int = MIN_DOCUMENT_TOKENS, scan_limit: int = 300) -> list[dict]:
    """Stream the test split and keep the first `n` articles over the length bar."""
    if CACHE.exists():
        return [json.loads(line) for line in CACHE.read_text().splitlines()][:n]

    from datasets import load_dataset

    stream = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
    kept = []
    for row in itertools.islice(stream, scan_limit):
        tokens = count_tokens(row["article"])
        if tokens < min_tokens:
            continue
        kept.append({"id": row["id"], "article": row["article"],
                     "reference": row["highlights"], "tokens": tokens})
        if len(kept) >= n:
            break
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text("\n".join(json.dumps(row) for row in kept))
    return kept
