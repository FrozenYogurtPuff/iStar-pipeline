# 0

import logging

from spacy.tokens import Span


# Bought [me] these books.
# -> me (dative, PRON / PROPN)
def dative_propn(s: Span):
    both: str = "Both"
    result = list()

    for token in s:
        if token.pos_ in ["PRON", "PROPN", "NOUN"] and token.dep_ == "dative":
            cur = (token, *token.conjuncts)
            for c in cur:
                result.append((c, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
