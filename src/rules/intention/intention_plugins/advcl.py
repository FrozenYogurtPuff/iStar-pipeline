# disapprove
import logging

from spacy.tokens import Span


def advcl(s: Span):
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "advcl":
            result.append((token, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
