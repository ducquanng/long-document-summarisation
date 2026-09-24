"""Splitting a document into pieces a model can actually read."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def chunk_by_tokens(sentences: Sequence[str], count_tokens: Callable[[str], int],
                    max_tokens: int) -> list[list[int]]:
    """Group consecutive sentences into chunks under the token budget.

    Returns indices rather than text so a caller can always recover the original
    position of every sentence, which is what keeps a summary in order.

    A sentence longer than the budget on its own becomes its own chunk: it will be
    truncated by the model, and that is worth knowing rather than hiding.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    running = 0
    for index, sentence in enumerate(sentences):
        length = count_tokens(sentence)
        if current and running + length > max_tokens:
            chunks.append(current)
            current, running = [], 0
        current.append(index)
        running += length
    if current:
        chunks.append(current)
    return chunks


def enforce_budget(chunks: list[list[int]], sentences: Sequence[str],
                   count_tokens: Callable[[str], int], max_tokens: int,
                   sort_within: bool = True) -> list[list[int]]:
    """Split any chunk that exceeds the budget.

    Clustering pays no attention to length, so a cluster can easily exceed what the
    encoder reads. Without this the overflow is silently truncated away.

    `sort_within` must stay false for any experiment about ordering: sorting here
    quietly restores document order inside every split chunk, which is exactly the
    variable such an experiment is trying to manipulate.
    """
    out: list[list[int]] = []
    for chunk in chunks:
        if count_tokens(" ".join(sentences[i] for i in chunk)) <= max_tokens:
            out.append(chunk)
            continue
        order = sorted(chunk) if sort_within else list(chunk)
        pieces = chunk_by_tokens([sentences[i] for i in order], count_tokens, max_tokens)
        out.extend([[order[i] for i in piece] for piece in pieces])
    return out
