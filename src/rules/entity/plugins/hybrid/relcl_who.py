# 14/17
import logging

from src.utils.spacy import token_not_end
from src.utils.typing import EntityRuleReturn, FixEntityLabel, SpacySpan


# [Auditors] who chase the dreams.
# Auditors -> chase (relcl)
def relcl_who(s: SpacySpan) -> EntityRuleReturn:
    actor: FixEntityLabel = "Actor"
    result = list()

    for token in s:
        if token.dep_ == "relcl":
            key = token.head
            if token_not_end(key) and key.nbor(1).lower_.startswith("who"):
                cur = (key, *key.conjuncts)
                for c in cur:
                    result.append((c, actor))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
