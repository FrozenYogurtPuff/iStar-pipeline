# slightly work

import logging

from spacy.tokens import Span

from src.utils.typing import RuleReturn


def xcomp_ask(s: Span) -> RuleReturn:
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "xcomp":
            head = token.head
            for child in head.children:
                if child.dep_ == "dobj":
                    result.append((child, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
