import logging

from spacy.tokens import Span

from src.utils.typing import RuleReturn


# All / The without any noun
def trailing_noun(s: Span) -> RuleReturn:
    no_label: str = "O"
    result = list()

    if s[-1].pos_ != "NOUN":
        result.append((s, no_label))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
