# 133/154
# [Resource] 123/154

import logging

from src.utils.typing import EntityRuleReturn, Span


# Anna's [home].
# home -> Anna (poss)
def poss(s: Span) -> EntityRuleReturn:
    resource: str = "Resource"
    result = list()

    for token in s:
        if token.dep_ == "poss":
            key = token
            result.append((key, resource))
            key_head = token.head
            result.append((key_head, resource))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
