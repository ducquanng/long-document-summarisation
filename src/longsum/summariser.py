"""Model wrapper, plus a fake one so the strategies can be tested without weights."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from .config import MODEL_MAX_INPUT, MODEL_NAME, SUMMARY_MAX_TOKENS, SUMMARY_MIN_TOKENS


class Summariser(Protocol):
    max_input_tokens: int

    def count_tokens(self, text: str) -> int: ...

    def summarise(self, text: str) -> str: ...


class Distilbart:
    """Encoder-decoder summarisation with an explicit generate call.

    The model reads at most `max_input_tokens`; anything beyond that is discarded
    before generation. Every strategy in this repository exists to decide *what* to
    put inside that window.
    """

    def __init__(self, model_name: str = MODEL_NAME, max_input_tokens: int = MODEL_MAX_INPUT,
                 num_beams: int = 1):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.num_beams = num_beams

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def summarise(self, text: str) -> str:
        import torch

        batch = self.tokenizer([text], return_tensors="pt", truncation=True,
                               max_length=self.max_input_tokens)
        with torch.no_grad():
            output = self.model.generate(
                **batch, min_length=SUMMARY_MIN_TOKENS, max_length=SUMMARY_MAX_TOKENS,
                num_beams=self.num_beams, no_repeat_ngram_size=3,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


class FirstSentences:
    """A deterministic stand-in: returns the opening sentences of its input.

    Lets the strategies, the budget logic and the ordering be tested for what they
    select, with no model download and no randomness.
    """

    def __init__(self, max_input_tokens: int = 100, n_sentences: int = 2):
        self.max_input_tokens = max_input_tokens
        self.n_sentences = n_sentences

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def summarise(self, text: str) -> str:
        from .chunking import split_sentences

        return " ".join(split_sentences(text)[: self.n_sentences])


@lru_cache(maxsize=2)
def _sentence_model(model_name: str):
    """Loaded once per process: rebuilding it per call dominated the runtime."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_sentences(sentences: Sequence[str], model_name: str = "all-MiniLM-L6-v2"):
    return _sentence_model(model_name).encode(list(sentences))
