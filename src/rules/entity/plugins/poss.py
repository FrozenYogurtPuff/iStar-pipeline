# 133/154
# [Resource] 123/154

import logging

from src.utils.typing import EntityRuleReturn, FixEntityLabel, SpacySpan


# Anna's [home].
# home -> Anna (poss)
def poss(s: SpacySpan) -> EntityRuleReturn:
    resource: FixEntityLabel = "Resource"
    result = list()

    for token in s:
        if token.dep_ == "poss":
            key = token.head
            cur = (key, *key.conjuncts)
            for c in cur:
                result.append((c, resource))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
