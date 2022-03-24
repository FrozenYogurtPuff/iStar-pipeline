# dep
# amod - Match 790 out of 1012 with 0.7806324110671937
# > dobj - 0.8958904109589041 - 327/365
# > nmod - 1.0 - 8/8
# > pobj - 0.801007556675063 - 318/397
# > poss - 0.8333333333333334 - 5/6
# appos - Match 69 out of 133 with 0.518796992481203
# > conj - 0.9090909090909091 - 10/11
# > nmod - 1.0 - 4/4
# cc - Match 195 out of 628 with 0.3105095541401274
# > nummod - 0.8333333333333334 - 5/6
# compound - Match 1380 out of 1937 with 0.7124419204956117
# > amod - 1.0 - 4/4
# > conj - 0.8201438848920863 - 114/139
# > dobj - 0.9288793103448276 - 431/464
# > nummod - 1.0 - 4/4
# conj - Match 404 out of 814 with 0.4963144963144963
# > nmod - 0.8709677419354839 - 27/31
# > nsubjpass - 0.8333333333333334 - 5/6
# > nummod - 0.8333333333333334 - 5/6
# dobj - Match 1127 out of 1469 with 0.7671885636487407
# > ccomp - 0.8229166666666666 - 79/96
# > pcomp - 0.8144329896907216 - 79/97
# > xcomp - 0.88 - 198/225
# nmod - Match 110 out of 134 with 0.8208955223880597
# > appos - 1.0 - 4/4
# > compound - 0.8461538461538461 - 11/13
# > conj - 0.875 - 7/8
# > dobj - 0.8780487804878049 - 36/41
# > pobj - 0.8205128205128205 - 32/39
# npadvmod - Match 25 out of 35 with 0.7142857142857143
# > amod - 0.9545454545454546 - 21/22
# nsubjpass - Match 98 out of 198 with 0.494949494949495
# > ccomp - 0.8125 - 13/16
# nummod - Match 33 out of 54 with 0.6111111111111112
# > dobj - 0.9090909090909091 - 10/11
# poss - Match 115 out of 144 with 0.7986111111111112
# > dobj - 0.9230769230769231 - 48/52
# predet - Match 13 out of 18 with 0.7222222222222222
# > dobj - 1.0 - 6/6
# punct - Match 318 out of 1984 with 0.16028225806451613
# > amod - 0.8055555555555556 - 29/36
# > compound - 0.84375 - 27/32

# dep.head
# amod - Match 815 out of 1012 with 0.8053359683794467
# > dobj - 0.9232876712328767 - 337/365
# > nmod - 1.0 - 8/8
# > pobj - 0.8236775818639799 - 327/397
# > poss - 0.8333333333333334 - 5/6
# appos - Match 68 out of 133 with 0.5112781954887218
# > conj - 0.9090909090909091 - 10/11
# > dobj - 0.8095238095238095 - 17/21
# > nmod - 1.0 - 4/4
# cc - Match 276 out of 628 with 0.4394904458598726
# > dobj - 0.8271604938271605 - 67/81
# > nmod - 0.8181818181818182 - 27/33
# > nummod - 0.8333333333333334 - 5/6
# compound - Match 1406 out of 1937 with 0.7258647392875581
# > amod - 1.0 - 4/4
# > compound - 0.803088803088803 - 208/259
# > conj - 0.8705035971223022 - 121/139
# > dobj - 0.9375 - 435/464
# > nsubjpass - 0.803921568627451 - 41/51
# > nummod - 1.0 - 4/4
# conj - Match 392 out of 814 with 0.48157248157248156
# > dobj - 0.8137254901960784 - 83/102
# > nmod - 0.8064516129032258 - 25/31
# > nsubjpass - 0.8333333333333334 - 5/6
# > nummod - 0.8333333333333334 - 5/6
# nmod - Match 113 out of 134 with 0.8432835820895522
# > appos - 1.0 - 4/4
# > compound - 0.8461538461538461 - 11/13
# > dobj - 0.8780487804878049 - 36/41
# > pobj - 0.8717948717948718 - 34/39
# npadvmod - Match 22 out of 35 with 0.6285714285714286
# > amod - 0.9545454545454546 - 21/22
# nummod - Match 34 out of 54 with 0.6296296296296297
# > dobj - 0.9090909090909091 - 10/11
# poss - Match 115 out of 144 with 0.7986111111111112
# > dobj - 0.9230769230769231 - 48/52
# predet - Match 13 out of 18 with 0.7222222222222222
# > dobj - 1.0 - 6/6
# punct - Match 518 out of 1984 with 0.2610887096774194
# > amod - 0.8055555555555556 - 29/36
# > compound - 0.84375 - 27/32
# > dep - 1.0 - 6/6
# > dobj - 0.8073394495412844 - 88/109


import logging

from spacy.tokens import Span

from src.utils.typing import EntityRuleReturn

logger = logging.getLogger(__name__)

resource: str = "Resource"


def dep_label(dep: str, head_dep: str) -> str | None:
    for dep_ans, head_dep_lists in [
        ("amod", ["dobj", "nmod", "pobj", "poss"]),
        ("appos", ["conj", "nmod"]),
        ("cc", ["nummod"]),
        ("compound", ["amod", "conj", "dobj", "nummod"]),
        ("conj", ["nmod", "nsubjpass", "nummod"]),
        ("dobj", ["ccomp", "pcomp", "xcomp"]),
        ("nmod", ["appos", "compound", "conj", "dobj", "pobj"]),
        ("npadvmod", ["amod"]),
        ("nsubjpass", ["ccomp"]),
        ("nummod", ["dobj"]),
        ("poss", ["dobj"]),
        ("predet", ["dobj"]),
        ("punct", ["amod", "compound"]),
    ]:
        if dep == dep_ans and head_dep in head_dep_lists:
            logger.debug(f"{dep}, {head_dep}")
            return resource
    return None


def dep_head_label(dep: str, head_dep: str) -> str | None:
    for dep_ans, head_dep_lists in [
        ("amod", ["dobj", "nmod", "pobj", "poss"]),
        ("appos", ["conj", "nmod", "dobj"]),
        ("cc", ["nummod", "nmod", "dobj"]),
        (
            "compound",
            ["amod", "compound", "conj", "dobj", "nsubjpass", "nummod"],
        ),
        ("conj", ["dobj", "nmod", "nsubjpass", "nummod"]),
        ("nmod", ["appos", "compound", "dobj", "pobj"]),
        ("npadvmod", ["amod"]),
        ("nummod", ["dobj"]),
        ("poss", ["dobj"]),
        ("predet", ["dobj"]),
        ("punct", ["amod", "compound", "dep", "dobj"]),
    ]:
        if dep == dep_ans and head_dep in head_dep_lists:
            logger.debug(f"Head: {dep}, {head_dep}")
            return resource
    return None


def dep_base(s: Span) -> EntityRuleReturn:
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
