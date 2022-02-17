import logging

from src.typing import SpacySpan, BertEntityLabelRaw, EntityRuleReturn


# Show things to [Anna].
# to (ADP, dative) -> Anna (pobj)
def dative_ADP(s: SpacySpan) -> EntityRuleReturn:
    actor: BertEntityLabelRaw = 'Actor'
    result = list()

    for token in s:
        if token.pos_ == 'ADP' and token.dep_ == 'dative':
            key = list(token.children)
            for k in key:
                if k.dep_ == 'pobj':
                    result.extend((k, *k.conjuncts))

    logging.getLogger(__name__).debug(f'Length {len(result)}: {result}')
    return result, actor
