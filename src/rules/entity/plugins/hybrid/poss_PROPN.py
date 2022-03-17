# 117/143
# PRON/PROPN 94/108
# 〃 [Resource] 89/108
# NOUN 28/35

import logging

from src.utils.typing import EntityRuleReturn, Span


# [Anna]'s home.
# -> Anna (poss, NOUN)
def poss_propn(s: Span) -> EntityRuleReturn:
    # 'Both' is a special case for both
    resource: str = "Resource"
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "poss":
            if token.pos_ in ["PRON", "PROPN"]:
                cur = (token, *token.conjuncts)
                for c in cur:
                    result.append((c, resource))
            elif token.pos_ in ["NOUN"]:
                cur = (token, *token.conjuncts)
                for c in cur:
                    result.append((c, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
