# ner
# WORK_OF_ART - Match 4 out of 5 with 0.8
# LAW - Match 1 out of 1 with 1.0
# LOC - Match 2 out of 2 with 1.0
# NORP - Match 3 out of 4 with 0.75
# EVENT - Match 2 out of 2 with 1.0
# FAC - Match 3 out of 4 with 0.75


import logging

from src.utils.typing import EntityRuleReturn, Span

logger = logging.getLogger(__name__)


def ner(s: Span) -> EntityRuleReturn:
    resource: str = "Resource"
    result = list()

    for ent in s.ents:
        if ent.label_ in ["WORK_OF_ART", "LAW", "LOC", "NORP", "EVENT", "FAC"]:
            logger.debug(f"{ent.label_}")
            result.append((ent, resource))

    logger.debug(f"Length {len(result)}: {result}")
    return result
