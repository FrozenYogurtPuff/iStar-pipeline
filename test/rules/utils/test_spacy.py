import spacy
import logging

from src.rules.utils.spacy import idx_valid
from src.utils.typing import SpacySpan, SpacyToken

logger = logging.getLogger(__name__)


def test_idx_valid():
    nlp: spacy.language.Language = spacy.load('en_core_web_lg')
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
