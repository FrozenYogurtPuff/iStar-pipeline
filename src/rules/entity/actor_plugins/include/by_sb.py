# no hit

import logging

from spacy.tokens import Span

from src.utils.typing import RuleReturn

logger = logging.getLogger(__name__)


# be able to
def by_sb(s: Span) -> RuleReturn:
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
