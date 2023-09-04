# no hit

import logging

from spacy.tokens import Span

logger = logging.getLogger(__name__)


# be able to
def by_sb(s: Span):
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "agent":
            for child in token.children:
                if child.dep_ == "pobj":
                    logger.debug(f"{token} {child}")
                    result.append((child, both))

    logger.debug(f"Length {len(result)}: {result}")
    return result
