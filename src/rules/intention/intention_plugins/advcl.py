# disapprove
import logging

from spacy.tokens import Span

from src.utils.typing import RuleReturn


def advcl(s: Span) -> RuleReturn:
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "advcl":
            result.append((token, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
