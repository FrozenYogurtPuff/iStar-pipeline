# dep
# case - Match 20 out of 49 (12 + 8) with 0.40816326530612246
#   > nsubj - 1.0 - 5+0/5
# compound - Match 843 out of 1937 (354 + 489) with 0.4352090862157976
#   > nsubj - 0.9550561797752809 - 121+219/356
#   > poss - 0.8181818181818182 - 2+7/11
# det - Match 1114 out of 2341 (444 + 670) with 0.4758650149508757
#   > nsubj - 0.9355704697986578 - 223+474/745
# nsubj - Match 954 out of 1197 (402 + 552) with 0.7969924812030075
#   ccomp - 0.7734375 - 99/128
# pobj - Match 487 out of 1715 (291 + 196) with 0.2839650145772595
#   agent - 0.75 - 45/60
#   dative - 0.7777777777777778 - 7/9

# dep.head
# case - Match 20 out of 49 (12 + 8) with 0.40816326530612246
#   > nsubj - 1.0 - 5+0/5
# compound - Match 831 out of 1937 (347 + 484) with 0.4290139390810532
#   > nsubj - 0.952247191011236 - 122+217/356
#   > poss - 0.9090909090909091 - 2+8/11
# det - Match 1136 out of 2341 (464 + 672) with 0.48526270824434004
#   > nsubj - 0.9476510067114094 - 233+473/745
# relcl - Match 51 out of 199 (42 + 9) with 0.2562814070351759
#   > appos - 0.875 - 14+0/16


import logging

from spacy.tokens import Span

logger = logging.getLogger(__name__)

both: str = "Both"
agent: str = "Agent"
role: str = "Role"


def dep_label(dep: str, head_dep: str) -> str | None:
    # if (dep, head_dep) in [
    #     ("compound", "nsubj"),
    #     ("compound", "poss"),
    #     ("det", "nsubj"),
    # ]:
    #     return both
    # elif (dep, head_dep) in []:
    #     return agent
    # elif (dep, head_dep) in [
    #     ("case", "agent"),
    # ]:
    #     return role

    return None


def dep_head_label(dep: str, head_dep: str) -> str | None:
    # if (dep, head_dep) in [
    #     ("compound", "nsubj"),
    #     ("det", "nsubj"),
    # ]:
    #     return both
    # elif (dep, head_dep) in []:
    #     return agent
    # elif (dep, head_dep) in [
    #     ("case", "nsubj"),
    # ]:
    #     return both

    return None


def dep_base(s: Span):
    result = list()

    for token in s:
        dep = token.dep_
        head_dep = token.head.dep_
        label = dep_label(dep, head_dep)
        if label:
            result.append((token, label))
        head_label = dep_head_label(dep, head_dep)
        if head_label:
            result.append((token.head, head_label))

    logger.debug(f"Length {len(result)}: {result}")

    return result
