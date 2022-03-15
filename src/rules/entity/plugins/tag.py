import logging
from typing import Optional

from src.utils.typing import EntityRuleReturn, FixEntityLabel, SpacySpan

logger = logging.getLogger(__name__)

both: FixEntityLabel = "Both"
resource: FixEntityLabel = "Resource"


def tag_label(tag: str) -> Optional[FixEntityLabel]:
    if tag in ["NNS", "NNP", "NNPS"]:
        return both

    # .85 ~ .9
    # if tag in ["PRP$", "POS", "NN", "HYPH", "DT"]:
    #     return both

    # .75 ~ .85
    # if tag in ["PDT"]:
    #     return both

    return None


def tag_conj_label(tag: str) -> Optional[FixEntityLabel]:
    if tag in ["NNPS"]:
        return both

    # .85 ~ .9
    # if tag in ["NN", "NNS"]:
    #     return both
    # if tag in ["CD", "JJR"]:
    #     return resource

    # .75 ~ .85
    # if tag in ["NNP", "JJ", "DT"]:
    #     return both

    return None


def tag_head_label(tag: str) -> Optional[FixEntityLabel]:
    if tag in ["HYPH"]:
        return both

    # .85 ~ .9
    # if tag in ["PDT", "DT"]:
    #     return both

    return None


def tag_base(s: SpacySpan) -> EntityRuleReturn:
    result = list()

    for token in s:
        tag = token.tag_
        label = tag_label(tag)
        if label:
            result.append((token, label))
        conj_label = tag_conj_label(tag)
        if conj_label:
            if not label:
                result.append((token, conj_label))
            for conj in token.conjuncts:
                result.append((conj, conj_label))
        head_label = tag_head_label(tag)
        if head_label:
            result.append((token.head, head_label))

    logger.debug(f"Length {len(result)}: {result}")

    return result
