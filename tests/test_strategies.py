import pytest

from longsum.chunking import chunk_by_tokens, enforce_budget, split_sentences
from longsum.strategies import clustered, lead, sequential
from longsum.summariser import FirstSentences

DOC = " ".join(f"Sentence number {i} carries a little information." for i in range(30))


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
def summariser():
    return FirstSentences(max_input_tokens=30, n_sentences=1)


def test_chunking_respects_the_budget(summariser):
    sentences = split_sentences(DOC)
    chunks = chunk_by_tokens(sentences, words, max_tokens=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert words(" ".join(sentences[i] for i in chunk)) <= 30 or len(chunk) == 1


def test_chunking_keeps_every_sentence_exactly_once():
    sentences = split_sentences(DOC)
    indices = [i for chunk in chunk_by_tokens(sentences, words, 30) for i in chunk]
    assert indices == list(range(len(sentences)))


def test_an_oversized_sentence_becomes_its_own_chunk():
    sentences = ["short one.", "word " * 100, "short two."]
    chunks = chunk_by_tokens(sentences, words, max_tokens=20)
    assert [1] in chunks


def test_enforce_budget_splits_an_oversized_group_and_keeps_order():
    sentences = split_sentences(DOC)
    oversized = [list(range(len(sentences)))]
    fixed = enforce_budget(oversized, sentences, words, max_tokens=30)
    assert len(fixed) > 1
    assert [i for group in fixed for i in group] == list(range(len(sentences)))


def test_sequential_summary_is_built_in_document_order(summariser):
    result = sequential(DOC, summariser)
    assert result.is_ordered
    assert result.n_chunks > 1


def test_lead_uses_no_model_and_takes_the_opening(summariser):
    result = lead(DOC, summariser, n_sentences=3)
    assert result.n_chunks == 0
    assert result.text.startswith("Sentence number 0")


def fake_embedder(sentences):
    """Two clusters by construction: even-indexed sentences against odd ones."""
    return [[0.0, float(i % 2)] for i, _ in enumerate(sentences)]


def test_clustering_with_order_preserved_reads_in_document_order(summariser):
    result = clustered(DOC, summariser, n_clusters=2, preserve_order=True, embedder=fake_embedder)
    assert result.strategy == "clustered_ordered"
    assert result.is_ordered
    for group in result.sentence_groups:
        assert group == sorted(group)


def test_clustering_without_order_scrambles_it(summariser):
    result = clustered(DOC, summariser, n_clusters=2, preserve_order=False, embedder=fake_embedder)
    assert result.strategy == "clustered_scrambled"
    # The second cluster starts at sentence 1, which comes before the first cluster's later
    # sentences: reading the groups in cluster order does not follow the document.
    flattened = [i for group in result.sentence_groups for i in group]
    assert flattened != sorted(flattened)


def test_budget_splitting_does_not_quietly_restore_order():
    """The bug this guards against made an ordering experiment measure nothing.

    Splitting an oversized cluster used to sort its sentences, which handed the
    scrambled variant its order back and made both variants score identically.
    """
    sentences = split_sentences(DOC)
    scrambled = [list(reversed(range(len(sentences))))]
    kept = enforce_budget(scrambled, sentences, words, max_tokens=30, sort_within=False)
    assert [i for group in kept for i in group] == list(reversed(range(len(sentences))))

    restored = enforce_budget(scrambled, sentences, words, max_tokens=30, sort_within=True)
    assert [i for group in restored for i in group] == list(range(len(sentences)))


def test_clustered_groups_respect_the_model_budget(summariser):
    sentences = split_sentences(DOC)
    result = clustered(DOC, summariser, n_clusters=2, preserve_order=True, embedder=fake_embedder)
    for group in result.sentence_groups:
        assert words(" ".join(sentences[i] for i in group)) <= summariser.max_input_tokens


def test_short_documents_fall_back_to_sequential(summariser):
    result = clustered("One. Two.", summariser, embedder=fake_embedder)
    assert result.strategy == "sequential"
