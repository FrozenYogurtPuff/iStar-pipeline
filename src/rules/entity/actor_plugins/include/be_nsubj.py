# no hit

import logging

from spacy.tokens import Span

from src.utils.spacy_utils import token_not_last


# be able to
def be_nsubj(s: Span):
    both: str = "Both"
    result = list()

    for token in s:
        if (
            token.lower_ == "be"
            and token_not_last(token)
            and token.nbor(1).lower_ == "able"
        ):
            for child in token.children:
                if child.dep_ == "nsubj":
                    result.append((child, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
