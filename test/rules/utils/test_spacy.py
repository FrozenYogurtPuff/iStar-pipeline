import logging
from collections.abc import Sequence

import spacy

from src.utils.spacy_utils import (
    get_bio_sent_from_char_spans,
    idx_valid,
    include_elem,
    token_not_first,
    token_not_last,
)

logger = logging.getLogger(__name__)


def test_idx_valid():
    nlp = spacy.load("en_core_web_trf")
    sent = nlp("Just Monika.")[:]
    assert not idx_valid(sent, -1)
    assert not idx_valid(sent, [0, -1])
    assert idx_valid(sent, 0)
    assert idx_valid(sent, 2)
    assert not idx_valid(sent, 3)
    assert idx_valid(sent, (0, 1, 2))
    assert not idx_valid(sent, 4)
    assert not idx_valid(sent, [0, 4])
    assert idx_valid(sent, 0, is_char=True)
    assert idx_valid(sent, 5, is_char=True)
    assert not idx_valid(sent, -1, is_char=True)
    assert not idx_valid(sent, 12, is_char=True)


def test_include_elem():
    nlp = spacy.load("en_core_web_trf")
    sent = nlp(
        "While the show is definitely cohesive, the seasons all manage to serve distinct purposes."
    )
    chunks = list(
        sent.noun_chunks
    )  # [the show, the seasons, distinct purposes]
    for c in chunks:
        assert include_elem(c, sent)
    assert include_elem(chunks[0], sent[1:3])
    assert include_elem(chunks[0][0], sent[1:3])
    assert not include_elem(chunks[0], sent[3:])
    assert include_elem(sent[1], chunks[0])


def test_get_bio_sent_from_char_spans():
    sent = "Hello Yuri! My name is Markov."
    labels: list[Sequence[int | str]] = [
        [0, 10, "Greeting"],
        [15, 22, "Extra"],
    ]
    result_token, result_label = get_bio_sent_from_char_spans(sent, labels)
    assert result_token == [
        "Hello",
        "Yuri",
        "!",
        "My",
        "name",
        "is",
        "Markov",
        ".",
    ]
    assert result_label == [
        "B-Greeting",
        "I-Greeting",
        "O",
        "O",
        "B-Extra",
        "I-Extra",
        "O",
        "O",
    ]

    sent = "Hello Yuri! My name is Markov."
    result_token, result_label = get_bio_sent_from_char_spans(sent)
    assert result_token == [
        "Hello",
        "Yuri",
        "!",
        "My",
        "name",
        "is",
        "Markov",
        ".",
    ]
    assert result_label == ["O", "O", "O", "O", "O", "O", "O", "O"]


def test_token_not_first():
    nlp = spacy.load("en_core_web_trf")
    sent = nlp(
        "What are some of the most underrated shows out there? We attempt to find the gems that might actually be worth your time."
    )
    first = True
    for tok in sent:
        if first:
            assert not token_not_first(tok)
            first = False
        else:
            assert token_not_first(tok)


def test_token_not_last():
    nlp = spacy.load("en_core_web_trf")
    sent = nlp(
        "And you don't seem to understand: a shame you seemed an honest man. And all the fears you hold so dear, will turn to whisper in your ear."
    )
    prev = None
    for tok in sent:
        if prev:
            assert token_not_last(prev)
        prev = tok
    assert not token_not_last(prev)
