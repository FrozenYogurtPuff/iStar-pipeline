# 66/73
# dative 7/8
# agent 59/65

import logging

from spacy.tokens import Span

# Show things to [Anna].
# to (ADP, dative) -> Anna (pobj)

# carried out by [immigrants].
# by (ADP, agent) -> immigrants (pobj)
def agent_dative_adp(s: Span):
    resource: str = "Resource"
    result = list()

    for token in s:
        if token.pos_ == "ADP" and token.dep_ in ["dative", "agent"]:
            key = list(token.children)
            for k in key:
                if k.dep_ == "pobj":
                    cur = (k, *k.conjuncts)
                    for c in cur:
                        result.append((c, resource))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
