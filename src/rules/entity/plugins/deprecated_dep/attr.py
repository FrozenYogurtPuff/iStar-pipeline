# 3/36

import logging

from src.utils.typing import EntityRuleReturn, Span


# attr
def attr(s: Span) -> EntityRuleReturn:
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "attr":
            key = token.head
            cur = (key, *key.conjuncts)
            for c in cur:
                result.append((c, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
