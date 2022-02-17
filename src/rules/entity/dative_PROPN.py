import logging

from src.typing import SpacySpan, BertEntityLabelRaw, EntityRuleReturn


# Bought [me] these books.
# -> me (dative, PRON / PROPN)
def dative_PROPN(s: SpacySpan) -> EntityRuleReturn:
    actor: BertEntityLabelRaw = 'Actor'
    result = list()

    for token in s:
        if token.pos_ in ['PRON', 'PROPN'] and token.dep_ == 'dative':
            result.extend((token, *token.conjuncts))

    logging.getLogger(__name__).debug(f'Length {len(result)}: {result}')
    return result, actor
