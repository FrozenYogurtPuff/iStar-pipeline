import logging

from spacy.tokens import Span

from src.utils.typing import EntityRuleReturn

logger = logging.getLogger(__name__)

both: str = "Both"
resource: str = "Resource"


def dep_label(dep: str) -> str | None:
    if dep in ["nmod", "compound"]:
        return both

    # .85 ~ .9
    # if dep in ["amod", "poss", "predet", "nsubj", "det", "case"]:
    #     return both

    # .75 ~ .85
    # if dep in ["npadvmod", "pobj", "dobj"]:
    #     return both

    return None


def dep_conj_label(dep: str) -> str | None:
    if dep in ["nsubjpass", "nmod"]:
        return both

    # .85 ~ .9
    # if dep in ["pobj", "dobj"]:
    #     return both

    # .75 ~ .85
    # if dep in ["nummod", "dep", "amod", "appos"]:
    #     return resource
    # if dep in ["nsubj"]:
    #     return both

    return None


def dep_head_label(dep: str) -> str | None:
    if dep in ["nmod", "compound", "appos"]:
        return both

    # .85 ~ .9
    # if dep in ["relcl", "poss", "predet", "det", "case", "amod"]:
    #     return both

    return None


def dep_base(s: Span) -> EntityRuleReturn:
    result = list()

    for token in s:
        dep = token.dep_
        label = dep_label(dep)
        if label:
            result.append((token, label))
        conj_label = dep_conj_label(dep)
        if conj_label:
            if not label:
                result.append((token, conj_label))
            for conj in token.conjuncts:
                result.append((conj, conj_label))
        head_label = dep_head_label(dep)
        if head_label:
            result.append((token.head, head_label))

    logger.debug(f"Length {len(result)}: {result}")

    return result
