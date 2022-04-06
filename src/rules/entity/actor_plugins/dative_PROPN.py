# 0

import logging

from spacy.tokens import Span

from src.utils.typing import RuleReturn


# Bought [me] these books.
# -> me (dative, PRON / PROPN)
def dative_propn(s: Span) -> RuleReturn:
    both: str = "Both"
    result = list()

    for token in s:
        if token.pos_ in ["PRON", "PROPN", "NOUN"] and token.dep_ == "actor":
            cur = (token, *token.conjuncts)
            for c in cur:
                result.append((c, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
