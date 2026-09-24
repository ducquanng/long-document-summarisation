import pytest

from longsum.rouge import all_scores, lcs_length, rouge_l, rouge_n, tokenize


def test_identical_text_scores_one():
    text = "the cat sat on the mat"
    assert rouge_n(text, text, 1).f1 == pytest.approx(1.0)
    assert rouge_n(text, text, 2).f1 == pytest.approx(1.0)
    assert rouge_l(text, text).f1 == pytest.approx(1.0)


def test_no_overlap_scores_zero():
    assert rouge_n("alpha beta", "gamma delta", 1).f1 == 0.0
    assert rouge_l("alpha beta", "gamma delta").f1 == 0.0


def test_rouge_1_matches_a_hand_calculation():
    """3 of 4 candidate tokens and 3 of 5 reference tokens overlap."""
    score = rouge_n("the cat sat here", "the cat sat on mats", 1)
    assert score.precision == pytest.approx(3 / 4)
    assert score.recall == pytest.approx(3 / 5)
    assert score.f1 == pytest.approx(2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5)))


def test_repeated_words_are_capped_at_the_reference_count():
    """Saying 'good' five times cannot earn credit for five matches."""
    score = rouge_n("good good good good good", "good results", 1)
    assert score.precision == pytest.approx(1 / 5)


def test_bigrams_notice_word_order_where_unigrams_do_not():
    unigram = rouge_n("sat cat the", "the cat sat", 1)
    bigram = rouge_n("sat cat the", "the cat sat", 2)
    assert unigram.f1 == pytest.approx(1.0)
    assert bigram.f1 == 0.0


def test_lcs_allows_gaps_but_not_reordering():
    assert lcs_length(["a", "b", "c"], ["a", "x", "b", "y", "c"]) == 3
    assert lcs_length(["c", "b", "a"], ["a", "b", "c"]) == 1


def test_rouge_l_prefers_the_ordered_summary():
    reference = "revenue rose sharply while costs fell"
    ordered = rouge_l("revenue rose while costs fell", reference).f1
    shuffled = rouge_l("costs fell revenue rose while", reference).f1
    assert ordered > shuffled


def test_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("The C.E.O.'s pay, 2024!") == ["the", "c", "e", "o", "s", "pay", "2024"]


def test_all_scores_returns_every_variant():
    assert set(all_scores("a b", "a b")) == {"rouge1", "rouge2", "rougeL"}


def test_rouge_1_cannot_see_the_order_pieces_are_joined_in():
    """Why the raw ordering comparison in the study is exactly zero.

    ROUGE-1 counts unigrams, so concatenating the same pieces in a different order
    gives an identical score. Any experiment about ordering that compares
    concatenated output on ROUGE-1 measures nothing, by construction.
    """
    reference = "the company raised guidance after a strong quarter in europe"
    pieces = ["a strong quarter in europe", "the company raised guidance"]
    forward = rouge_n(" ".join(pieces), reference, 1).f1
    backward = rouge_n(" ".join(reversed(pieces)), reference, 1).f1
    assert forward == backward

    # ROUGE-L is sensitive to it, because a subsequence has to be in order.
    assert rouge_l(" ".join(pieces), reference).f1 != rouge_l(" ".join(reversed(pieces)), reference).f1
