# 117/143
# PRON/PROPN 94/108
# 〃 [Resource] 89/108
# TODO: 对于 PRON/PROPN 的 Resource，其实捕捉到的是 their 等人称，需要之后追加 merge noun chunks
# NOUN 28/35

import logging

from src.utils.typing import SpacySpan, FixEntityLabel, EntityRuleReturn


# [Anna]'s home.
# -> Anna (poss, NOUN)
def poss_PROPN(s: SpacySpan) -> EntityRuleReturn:
    # 'Both' is a special case for both
    resource: FixEntityLabel = 'Resource'
    both: FixEntityLabel = 'Both'
    result = list()

    for token in s:
        if token.dep_ == 'poss':
            if token.pos_ in ['PRON', 'PROPN']:
                cur = (token, *token.conjuncts)
                for c in cur:
                    result.append((c, resource))
            elif token.pos_ in ['NOUN']:
                cur = (token, *token.conjuncts)
                for c in cur:
                    result.append((c, both))

    logging.getLogger(__name__).debug(f'Length {len(result)}: {result}')
    return result
