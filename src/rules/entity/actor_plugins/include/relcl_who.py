# no hit

import logging

from spacy.tokens import Span

from src.utils.spacy_utils import token_not_last


# [Auditors] who chase the dreams.
# Auditors -> chase (relcl)
def relcl_who(s: Span):
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "relcl":
            key = token.head
            if token_not_last(key) and key.nbor(1).lower_.startswith("who"):
                cur = (key, *key.conjuncts)
                for c in cur:
                    result.append((c, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
