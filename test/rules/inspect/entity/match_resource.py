import logging
from collections import Counter
from test.rules.inspect.utils import cache_nlp, dep_list
from test.rules.utils.load_dataset import load_dataset
from typing import Literal

from src.utils.spacy import char_idx_to_word, get_spacy, include_elem

logger = logging.getLogger(__name__)


class LabelTypeException(Exception):
    pass


if __name__ == "__main__":
    data = list(load_dataset("pretrained_data/2022/resource/all.jsonl"))

    nlp = get_spacy()

    resource: str = "Resource"

    mode: Literal["dep", "tag", "ner"] = "dep"
    check_list = eval(f"{mode}_list")
    include: Literal["", "conj", "head"] = "head"

    for check in check_list:
        # check head
        total_list = list()
        resource_list = list()

        total, match = 0, 0
        match_resource = 0
        for i, sent, anno in data:
            sent_processed = cache_nlp(nlp, sent)

            if mode == "ner":
                traverse_list = sent_processed.ents
            else:
                traverse_list = sent_processed

            for token in traverse_list:
                if mode == "tag":
                    key = token.tag_
                elif mode == "dep":
                    key = token.dep_
                elif mode == "ner":
                    key = token.label_
                else:
                    raise LabelTypeException(f"Unrecognized type {mode}")

                if key == check:
                    if include == "conj":
                        targets = token.conjuncts
                    elif include == "head":
                        targets = [token.head]
                    elif include == "":
                        targets = [token]
                    else:
                        raise LabelTypeException(
                            f"Unrecognized type {include}"
                        )

                    total += len(targets)
                    total_list.append(token.head.dep_)

                    for start, end, lab in anno:
                        ground = char_idx_to_word(sent_processed, start, end)

                        for target in targets:
                            if include_elem(target, ground):

                                match += 1
                                if lab == resource:
                                    match_resource += 1
                                    resource_list.append(token.head.dep_)
                                else:
                                    raise LabelTypeException(
                                        f"Illegal label type {lab}"
                                    )
                                break

        if total > 0 and match / total > 0.4:
            logger.error(
                f"{check} - Match {match} out of {total} with {match / total if total else 0}"
            )
        else:
            logger.warning(
                f"{check} - Match {match} out of {total} with {match / total if total else 0}"
            )

        total_counter = Counter(total_list)
        resource_counter = Counter(resource_list)
        for dep_key in dep_list:
            if dep_key in total_counter:
                target = resource_counter[dep_key]
                prob = target / total_counter[dep_key]
                if prob > 0.8 and target > 3:
                    logger.warning(
                        f"> {dep_key} - {prob} - {target}/{total_counter[dep_key]}"
                    )

# ner
# WORK_OF_ART - Match 4 out of 5 with 0.8
# LAW - Match 1 out of 1 with 1.0
# LOC - Match 2 out of 2 with 1.0
# NORP - Match 3 out of 4 with 0.75
# EVENT - Match 2 out of 2 with 1.0
# FAC - Match 3 out of 4 with 0.75

# tag
# CC - Match 197 out of 626 with 0.3146964856230032
# > nummod - 0.8333333333333334 - 5/6
# CD - Match 41 out of 64 with 0.640625
# > dobj - 0.9090909090909091 - 10/11
# HYPH - Match 69 out of 103 with 0.6699029126213593
# > amod - 0.8928571428571429 - 25/28
# > compound - 0.8421052631578947 - 16/19
# > dobj - 1.0 - 4/4
# > nmod - 0.8181818181818182 - 9/11
# JJ - Match 698 out of 1155 with 0.6043290043290044
# > amod - 0.9523809523809523 - 20/21
# > compound - 0.8333333333333334 - 40/48
# > dobj - 0.8974358974358975 - 280/312
# > nmod - 1.0 - 10/10
# > poss - 0.8571428571428571 - 6/7
# NN - Match 2528 out of 4086 with 0.6186979931473323
# > acl - 0.8666666666666667 - 52/60
# > amod - 0.9230769230769231 - 24/26
# > compound - 0.8888888888888888 - 112/126
# > conj - 0.8086642599277978 - 224/277
# > dobj - 0.9254385964912281 - 422/456
# > nmod - 0.8214285714285714 - 23/28
# > pobj - 0.8077753779697624 - 374/463
# > xcomp - 0.8315789473684211 - 79/95
# NNP - Match 599 out of 1256 with 0.4769108280254777
# > dobj - 0.8641975308641975 - 70/81
# NNPS - Match 54 out of 61 with 0.8852459016393442
# > pcomp - 1.0 - 4/4
# NNS - Match 1229 out of 1670 with 0.7359281437125749
# > advcl - 0.8775510204081632 - 43/49
# > appos - 0.8181818181818182 - 9/11
# > compound - 0.9166666666666666 - 11/12
# > conj - 0.8366013071895425 - 128/153
# > dobj - 0.8314606741573034 - 74/89
# > pcomp - 0.9166666666666666 - 44/48
# > relcl - 0.8235294117647058 - 28/34
# > xcomp - 0.944 - 118/125
# PDT - Match 13 out of 19 with 0.6842105263157895
# > dobj - 0.8571428571428571 - 6/7
# PRP$ - Match 86 out of 103 with 0.8349514563106796
# > dobj - 0.9130434782608695 - 42/46
# > pobj - 0.8478260869565217 - 39/46
# SYM - Match 39 out of 62 with 0.6290322580645161
# > compound - 0.8461538461538461 - 11/13
# > dobj - 0.8888888888888888 - 8/9
# VBG - Match 72 out of 335 with 0.21492537313432836
# > nmod - 1.0 - 5/5
# VBN - Match 139 out of 530 with 0.2622641509433962
# > compound - 1.0 - 4/4

# tag.head
# CC - Match 273 out of 626 with 0.43610223642172524
# > dobj - 0.8181818181818182 - 63/77
# > nmod - 0.8181818181818182 - 27/33
# > nummod - 0.8333333333333334 - 5/6
# CD - Match 38 out of 64 with 0.59375
# > dobj - 0.9090909090909091 - 10/11
# HYPH - Match 70 out of 103 with 0.6796116504854369
# > amod - 0.8928571428571429 - 25/28
# > compound - 0.8421052631578947 - 16/19
# > dobj - 1.0 - 4/4
# > nmod - 0.8181818181818182 - 9/11
# > pobj - 0.8571428571428571 - 6/7
# JJ - Match 706 out of 1155 with 0.6112554112554113
# > amod - 0.9523809523809523 - 20/21
# > compound - 0.8333333333333334 - 40/48
# > dobj - 0.9294871794871795 - 290/312
# > nmod - 1.0 - 10/10
# > pobj - 0.8042813455657493 - 263/327
# > poss - 0.8571428571428571 - 6/7
# NN - Match 1473 out of 4086 with 0.3604992657856094
# > amod - 0.9615384615384616 - 25/26
# > compound - 0.9206349206349206 - 116/126
# > dobj - 0.9342105263157895 - 426/456
# > nsubjpass - 0.8181818181818182 - 36/44
# > pobj - 0.8099352051835853 - 375/463
# NNP - Match 501 out of 1256 with 0.39888535031847133
# > dobj - 0.8641975308641975 - 70/81
# NNS - Match 427 out of 1670 with 0.255688622754491
# > appos - 0.9090909090909091 - 10/11
# > compound - 1.0 - 12/12
# > dobj - 0.8314606741573034 - 74/89
# PDT - Match 14 out of 19 with 0.7368421052631579
# > dobj - 1.0 - 7/7
# PRP$ - Match 86 out of 103 with 0.8349514563106796
# > dobj - 0.9130434782608695 - 42/46
# > pobj - 0.8478260869565217 - 39/46
# SYM - Match 38 out of 62 with 0.6129032258064516
# > compound - 0.8461538461538461 - 11/13
# > dobj - 0.8888888888888888 - 8/9
# VBG - Match 106 out of 335 with 0.3164179104477612
# > dobj - 0.8222222222222222 - 37/45
# > nmod - 1.0 - 5/5
# VBN - Match 232 out of 530 with 0.4377358490566038
# > compound - 1.0 - 4/4
# > dobj - 0.8761904761904762 - 92/105
# > pobj - 0.8487394957983193 - 101/119
# VBP - Match 34 out of 143 with 0.23776223776223776
# > dobj - 1.0 - 13/13
# VBZ - Match 42 out of 367 with 0.11444141689373297
# > dobj - 0.8421052631578947 - 16/19

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
