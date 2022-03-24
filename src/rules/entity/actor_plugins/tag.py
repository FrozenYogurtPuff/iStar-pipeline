# tag
# DT - Match 1118 out of 2371 (448 + 670) with 0.471530999578237
#   > nsubj - 0.9368279569892473 - 223+474/744
# HYPH - Match 34 out of 102 (9 + 25) with 0.3333333333333333
#   > nsubj - 1.0 - 2+12/14
# NN - Match 1303 out of 4081 (673 + 630) with 0.31928448909580986
#   > nsubj - 0.8230088495575221 - 59+34/113
# NNP - Match 950 out of 1255 (268 + 682) with 0.7569721115537849
#   > acl - 0.8823529411764706 - 1+14/17
#   > nsubj - 0.9921259842519685 - 66+186/254
#   > nsubjpass - 0.8333333333333334 - 1+9/12
# NNS - Match 411 out of 1667 (350 + 61) with 0.2465506898620276
#   > agent - 0.8333333333333334 - 9+1/12
#   > poss - 1.0 - 0+4/4
# POS - Match 19 out of 48 (10 + 9) with 0.3958333333333333
#   > nsubj - 1.0 - 5+0/5

# tag.head
# DT - Match 1138 out of 2371 (466 + 672) with 0.4799662589624631
#   > nsubj - 0.9489247311827957 - 233+473/744
# HYPH - Match 36 out of 102 (10 + 26) with 0.35294117647058826
#   > nsubj - 1.0 - 2+12/14
# NN - Match 422 out of 4081 (303 + 119) with 0.10340602793432982
#   > nsubj - 0.8407079646017699 - 63+32/113
# NNP - Match 659 out of 1255 (223 + 436) with 0.5250996015936255
#   > acl - 0.8235294117647058 - 1+13/17
#   > nsubj - 0.9921259842519685 - 66+186/254
#   > nsubjpass - 0.8333333333333334 - 3+7/12
# NNS - Match 103 out of 1667 (78 + 25) with 0.0617876424715057
#   > poss - 1.0 - 0+4/4
# POS - Match 18 out of 48 (10 + 8) with 0.375
#   > nsubj - 1.0 - 5+0/5
# VBZ - Match 22 out of 367 (18 + 4) with 0.05994550408719346
#   > appos - 0.9 - 9+0/10


import logging

from spacy.tokens import Span

from src.utils.typing import EntityRuleReturn

logger = logging.getLogger(__name__)

both: str = "Both"
agent: str = "Agent"
role: str = "Role"


def tag_label(tag: str, head_dep: str) -> str | None:
    # if (tag, head_dep) in [
    #     ("DT", "nsubj"),      # NG
    #     ("HYPH", "nsubj"),    # no hit
    #     ("NNP", "nsubj"),     # soso
    # ]:
    #     return both
    # elif (tag, head_dep) in [
    #     ("NNP", "acl"),         # soso
    #     ("NNS", "poss"),        # no hit
    # ]:
    #     return agent
    # elif (tag, head_dep) in [("POS", "nsubj")]:   # just 's, soso
    #     return role

    return None


def tag_head_label(tag: str, head_dep: str) -> str | None:
    # if (tag, head_dep) in [
    #     ("DT", "nsubj"),        # soso, caused mainly BERT-classification failed
    #     ("HYPH", "nsubj"),        # no hit
    #     ("NNP", "nsubj"),        # few
    # ]:
    #     return both
    # elif (tag, head_dep) in [
    #     ("NNS", "poss"),            # no hit
    # ]:
    #     return agent
    # elif (tag, head_dep) in [
    #     ("POS", "nsubj"),    # no hit
    #     ("VBZ", "appos"),    # no hit
    # ]:
    #     return role

    return None


def tag_base(s: Span) -> EntityRuleReturn:
    result = list()

    for token in s:
        tag = token.tag_
        head_dep = token.head.dep_
        label = tag_label(tag, head_dep)
        if label:
            result.append((token, label))
        head_label = tag_head_label(tag, head_dep)
        if head_label:
            result.append((token.head, head_label))

    logger.debug(f"Length {len(result)}: {result}")

    return result
