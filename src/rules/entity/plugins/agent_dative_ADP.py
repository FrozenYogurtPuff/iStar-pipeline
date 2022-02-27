# 66/73
# dative 7/8
# agent 59/65

import logging

from src.utils.typing import EntityRuleReturn, FixEntityLabel, SpacySpan

# Show things to [Anna].
# to (ADP, dative) -> Anna (pobj)

# carried out by [immigrants].
# by (ADP, agent) -> immigrants (pobj)
def agent_dative_adp(s: SpacySpan) -> EntityRuleReturn:
    both: FixEntityLabel = "Both"
    result = list()

    for token in s:
        if token.pos_ == "ADP" and token.dep_ in ["dative", "agent"]:
            key = list(token.children)
            for k in key:
                if k.dep_ == "pobj":
                    cur = (k, *k.conjuncts)
                    for c in cur:
                        result.append((c, both))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
