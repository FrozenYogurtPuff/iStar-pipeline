import logging

from spacy.tokens import Span


# TODO: 这个好像没用了吧，考虑去掉
# All / The without any noun
def trailing_noun(s: Span):
    no_label: str = "O"
    result = list()

    if s[-1].pos_ != "NOUN":
        result.append((s, no_label))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
