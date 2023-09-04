# ner
# GPE - Match 19 out of 23 (2 + 17) with 0.8260869565217391
# ORG - Match 242 out of 334 (75 + 167) with 0.7245508982035929
# PERSON - Match 7 out of 10 (0 + 7) with 0.7


import logging

from src.utils.typing import Span

logger = logging.getLogger(__name__)


def ner(s: Span):
    both: str = "Both"
    agent: str = "Agent"
    result = list()

    for ent in s.ents:
        # if ent.label_ in ["PERSON"]:
        #     result.append((ent, both))
        if ent.label_ in [
            # "ORG",
            # "GPE"
        ]:
            result.append((ent, both))

    logger.debug(f"Length {len(result)}: {result}")
    return result
