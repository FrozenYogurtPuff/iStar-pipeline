# 2866/3498
# dobj 1320/1595
# 〃 [Resource] 1223/1595
# pobj 1546/1903
# 〃 [Resource] 1231/1903
# Hybrid: dobj [Resource] + pobj 2769/3498

import logging

from src.utils.typing import EntityRuleReturn, FixEntityLabel, SpacySpan


# dobj, pobj
def dobj_pobj(s: SpacySpan) -> EntityRuleReturn:
    both: FixEntityLabel = "Both"
    resource: FixEntityLabel = "Resource"
    result = list()

    for token in s:
        if token.dep_ in ["dobj", "pobj"]:
            cur = (token, *token.conjuncts)
            label = resource if token.dep_ == "dobj" else both
            for c in cur:
                result.append((c, label))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
