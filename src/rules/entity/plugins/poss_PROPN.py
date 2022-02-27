# 117/143
# PRON/PROPN 94/108
# 〃 [Resource] 89/108
# NOUN 28/35

import logging

from src.utils.typing import EntityRuleReturn, FixEntityLabel, SpacySpan


# [Anna]'s home.
# -> Anna (poss, NOUN)
def poss_propn(s: SpacySpan) -> EntityRuleReturn:
    # 'Both' is a special case for both
    resource: FixEntityLabel = "Resource"
    both: FixEntityLabel = "Both"
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
