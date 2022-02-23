# ORG 317/334
# 〃 [Actor] 245/334
# NORP 4/4
# PERSON 10/10
# 〃 [Actor] 7/10
# PRODUCT 11/12
# 〃 [Actor] 8/12
# GPE 22/23
# 〃 [Actor] 19/23
# LOC 2/2
# WORK_OF_ART 5/5
# FAC 4/4
# EVENT 2/2
# LAW 1/1
# HYBRID: 368/396


import logging

from src.utils.typing import SpacySpan, FixEntityLabel, EntityRuleReturn

logger = logging.getLogger(__name__)


def ner(s: SpacySpan) -> EntityRuleReturn:
    actor: FixEntityLabel = 'Actor'
    both: FixEntityLabel = 'Both'
    result = list()

    for ent in s.ents:
        if ent.label_ in ['ORG', 'NORP', 'LOC', 'WORK_OF_ART', 'FAC', 'EVENT']:
            result.append((ent[0], both))
        elif ent.label_ in ['PERSON', 'PRODUCT', 'GPE']:
            result.append((ent[0], actor))

    logger.debug(f'Length {len(result)}: {result}')
    return result
