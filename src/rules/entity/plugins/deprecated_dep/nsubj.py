# 85/1341

import logging

from src.utils.typing import EntityRuleReturn, Span


# nsubj
def nsubj(s: Span) -> EntityRuleReturn:
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "nsubj":
            key = token
            result.append((key, both))
            key_conjuncts = token.conjuncts
            for conj in key_conjuncts:
                result.append((conj, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
