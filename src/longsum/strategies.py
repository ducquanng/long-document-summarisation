"""Four ways to summarise a document that does not fit in the window.

Each returns a summary and the sentence indices it was built from, so a result can
be traced back to the text it came from and the ordering can be checked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .chunking import chunk_by_tokens, enforce_budget, split_sentences
from .summariser import Summariser


@dataclass
class SummaryResult:
    strategy: str
    text: str
    n_chunks: int
    sentence_groups: list[list[int]] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    @property
    def is_ordered(self) -> bool:
        """True when the pieces were summarised in the order they appear in the document."""
        firsts = [min(g) for g in self.sentence_groups if g]
        return firsts == sorted(firsts)


def truncate(document: str, summariser: Summariser) -> SummaryResult:
    """The do-nothing baseline: hand over the document and let the model cut it off.

    Worth measuring rather than assuming. If nothing beats this, the machinery is
    not earning its runtime.
    """
    sentences = split_sentences(document)
    return SummaryResult("truncate", summariser.summarise(document), 1, [list(range(len(sentences)))])


def lead(document: str, summariser: Summariser, n_sentences: int = 3) -> SummaryResult:
    """Extractive baseline: the first few sentences, no model involved.

    In news writing the opening carries the story, which is why this baseline is
    famously hard to beat and belongs in any summarisation comparison.
    """
    sentences = split_sentences(document)
    picked = sentences[:n_sentences]
    return SummaryResult("lead", " ".join(picked), 0, [list(range(len(picked)))])


def sequential(document: str, summariser: Summariser) -> SummaryResult:
    """Chunk in reading order, summarise each chunk, join the summaries."""
    sentences = split_sentences(document)
    groups = chunk_by_tokens(sentences, summariser.count_tokens, summariser.max_input_tokens)
    parts = [summariser.summarise(" ".join(sentences[i] for i in group)) for group in groups]
    return SummaryResult("sequential", " ".join(parts), len(groups), groups)


def clustered(document: str, summariser: Summariser, n_clusters: int | None = None,
              preserve_order: bool = True,
              embedder: Callable[[Sequence[str]], object] | None = None,
              seed: int = 42) -> SummaryResult:
    """Group semantically similar sentences, then summarise each group.

    The idea is that a document returns to the same subject in several places, so
    grouping by meaning gives each summary a coherent subject instead of a slice of
    the timeline.

    `preserve_order` is the experiment. With it off - the way the original
    coursework did it - sentences are fed to the model in cluster order, which
    scrambles the narrative. With it on, sentences keep their document order inside
    each cluster and the clusters are processed in order of first appearance.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    sentences = split_sentences(document)
    if len(sentences) < 4:
        return sequential(document, summariser)

    k = n_clusters or max(2, min(len(sentences) // 10, 8))
    embed = embedder or (lambda s: __import__("longsum.summariser", fromlist=["x"]).embed_sentences(s))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(np.asarray(embed(sentences)))

    groups = [[i for i, label in enumerate(labels) if label == c] for c in range(k)]
    groups = [g for g in groups if g]
    if preserve_order:
        groups = [sorted(g) for g in groups]
        groups.sort(key=min)
    groups = enforce_budget(groups, sentences, summariser.count_tokens,
                            summariser.max_input_tokens, sort_within=preserve_order)
    if preserve_order:
        groups.sort(key=min)

    parts = [summariser.summarise(" ".join(sentences[i] for i in group)) for group in groups]
    name = "clustered_ordered" if preserve_order else "clustered_scrambled"
    return SummaryResult(name, " ".join(parts), len(groups), groups)


STRATEGIES = {
    "truncate": truncate,
    "lead": lead,
    "sequential": sequential,
    "clustered_ordered": lambda d, s: clustered(d, s, preserve_order=True),
    "clustered_scrambled": lambda d, s: clustered(d, s, preserve_order=False),
}


def reduce_pass(result: SummaryResult, summariser: Summariser) -> SummaryResult:
    """Summarise the joined chunk summaries once more, into a single short summary.

    Without this, a strategy that produces more chunks produces a longer final text,
    and ROUGE F1 punishes length through precision. Comparing a five-chunk strategy
    with a one-chunk strategy on raw output is therefore comparing lengths, not
    methods. The reduce pass puts every strategy on the same footing.
    """
    if result.n_chunks <= 1:
        return SummaryResult(f"{result.strategy}_reduced", result.text,
                             result.n_chunks, result.sentence_groups)
    return SummaryResult(
        f"{result.strategy}_reduced",
        summariser.summarise(result.text),
        result.n_chunks,
        result.sentence_groups,
    )
