# 4/8
# PRON/PROPN 0/4
# NOUN 4/4

import logging

from src.typing import SpacySpan, FixEntityLabel, EntityRuleReturn


# Bought [me] these books.
# -> me (dative, PRON / PROPN)
def dative_PROPN(s: SpacySpan) -> EntityRuleReturn:
    # 'Both' is a special case for both
    actor: FixEntityLabel = 'Actor'
    result = list()

    for token in s:
        if token.pos_ in ['PRON', 'PROPN', 'NOUN'] and token.dep_ == 'actor':
            cur = (token, *token.conjuncts)
            for c in cur:
                result.append((c, actor))

    logging.getLogger(__name__).debug(f'Length {len(result)}: {result}')
    return result
