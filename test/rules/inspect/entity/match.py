import logging
from test.rules.inspect.utils import cache_nlp, tag_list
from test.rules.utils.load_dataset import load_dataset

from src.utils.spacy import char_idx_to_word, get_spacy, include_elem

logger = logging.getLogger(__name__)


class LabelTypeException(Exception):
    pass


if __name__ == "__main__":
    data = list(load_dataset("pretrained_data/entity_ar_r_combined/all.jsonl"))

    nlp = get_spacy()

    actor: str = "Actor"
    resource: str = "Resource"

    # for check in dep_list:
    for check in tag_list:
        # for check in ner_list:
        total, match = 0, 0
        match_resource, match_actor = 0, 0
        for i, sent, anno in data:
            sent_processed = cache_nlp(nlp, sent)
            for token in sent_processed:
                # for token in sent_processed.ents:
                if token.tag_ == check:
                    # if token.label_ == check:
                    token_conjuncts = token.conjuncts
                    total += len(token_conjuncts)
                    for conj in token_conjuncts:
                        for start, end, lab in anno:
                            ground = char_idx_to_word(
                                sent_processed, start, end
                            )
                            if include_elem(conj, ground):
                                # if include_elem(token, ground):
                                # if include_elem(token.head, ground):
                                match += 1
                                if lab == actor:
                                    match_actor += 1
                                elif lab == resource:
                                    match_resource += 1
                                else:
                                    raise LabelTypeException(
                                        f"Illegal label type {lab}"
                                    )
                                break

        if total > 0 and match / total > 0.4:
            logger.error(
                f"{check} - Match {match} out of {total} ({match_actor} + {match_resource}) with {match / total if total else 0}"
            )
        else:
            logger.warning(
                f"{check} - Match {match} out of {total} ({match_actor} + {match_resource}) with {match / total if total else 0}"
            )

# dep_
# amod - Match 870 out of 1011 (104 + 766) with 0.8605341246290801
# appos - Match 94 out of 133 (32 + 62) with 0.706766917293233
# poss - Match 125 out of 144 (15 + 110) with 0.8680555555555556
# predet - Match 16 out of 18 (3 + 13) with 0.8888888888888888
# pobj - Match 1390 out of 1709 (389 + 1001) with 0.8133411351667642
# nsubjpass - Match 143 out of 196 (65 + 78) with 0.7295918367346939
# nsubj - Match 1027 out of 1198 (907 + 120) with 0.8572621035058431
# nmod - Match 123 out of 134 (34 + 89) with 0.917910447761194
# npadvmod - Match 27 out of 35 (3 + 24) with 0.7714285714285715
# dobj - Match 1211 out of 1464 (117 + 1094) with 0.8271857923497268
# det - Match 2012 out of 2330 (1015 + 997) with 0.863519313304721
# compound - Match 1822 out of 1928 (668 + 1154) with 0.9450207468879668
# case - Match 44 out of 49 (16 + 28) with 0.8979591836734694

# dep_.conjuncts
# pobj - Match 166 out of 194 (22 + 144) with 0.8556701030927835
# nummod - Match 5 out of 6 (0 + 5) with 0.8333333333333334
# nsubj - Match 15 out of 20 (11 + 4) with 0.75
# nsubjpass - Match 10 out of 10 (4 + 6) with 1.0
# nmod - Match 31 out of 32 (5 + 26) with 0.96875
# dobj - Match 114 out of 131 (10 + 104) with 0.8702290076335878
# dep - Match 3 out of 4 (0 + 3) with 0.75
# det - Match 1 out of 1 (0 + 1) with 1.0
# amod - Match 25 out of 31 (1 + 24) with 0.8064516129032258
# appos - Match 34 out of 43 (2 + 32) with 0.7906976744186046

# dep_.head
# relcl - Match 175 out of 199 (42 + 133) with 0.8793969849246231
# poss - Match 125 out of 144 (15 + 110) with 0.8680555555555556
# preconj - Match 5 out of 6 (2 + 3) with 0.8333333333333334
# predet - Match 16 out of 18 (3 + 13) with 0.8888888888888888
# nmod - Match 122 out of 134 (31 + 91) with 0.9104477611940298
# det - Match 2052 out of 2330 (1033 + 1019) with 0.8806866952789699
# compound - Match 1838 out of 1928 (658 + 1180) with 0.9533195020746889
# case - Match 44 out of 49 (16 + 28) with 0.8979591836734694
# appos - Match 121 out of 133 (63 + 58) with 0.9097744360902256
# amod - Match 898 out of 1011 (108 + 790) with 0.8882294757665677

# tag_
# PRP$ - Match 90 out of 103 (6 + 84) with 0.8737864077669902
# POS - Match 43 out of 48 (15 + 28) with 0.8958333333333334
# PDT - Match 16 out of 19 (3 + 13) with 0.8421052631578947
# NNS - Match 1532 out of 1662 (339 + 1193) with 0.9217809867629362
# NNP - Match 1195 out of 1254 (823 + 372) with 0.9529505582137161
# NNPS - Match 61 out of 61 (12 + 49) with 1.0
# NN - Match 3467 out of 4073 (1146 + 2321) with 0.8512153204026516
# LS - Match 1 out of 1 (0 + 1) with 1.0
# JJR - Match 11 out of 15 (2 + 9) with 0.7333333333333333
# HYPH - Match 90 out of 103 (28 + 62) with 0.8737864077669902
# DT - Match 2025 out of 2360 (1019 + 1006) with 0.8580508474576272

# tag_.conjuncts
# NN - Match 601 out of 707 (55 + 546) with 0.8500707213578501
# NNP - Match 113 out of 143 (16 + 97) with 0.7902097902097902
# NNPS - Match 8 out of 8 (2 + 6) with 1.0
# NNS - Match 429 out of 485 (57 + 372) with 0.8845360824742268
# JJ - Match 57 out of 75 (3 + 54) with 0.76
# JJR - Match 6 out of 7 (0 + 6) with 0.8571428571428571
# FW - Match 14 out of 19 (0 + 14) with 0.7368421052631579
# CD - Match 7 out of 8 (0 + 7) with 0.875
# DT - Match 5 out of 6 (2 + 3) with 0.8333333333333334

# tag_.head
# PDT - Match 17 out of 19 (3 + 14) with 0.8947368421052632
# HYPH - Match 93 out of 103 (30 + 63) with 0.9029126213592233
# DT - Match 2054 out of 2360 (1035 + 1019) with 0.8703389830508474

# ents
# CARDINAL - Match 27 out of 36 (2 + 25) with 0.75
# EVENT - Match 2 out of 2 (0 + 2) with 1.0
# FAC - Match 4 out of 4 (3 + 1) with 1.0
# GPE - Match 22 out of 23 (17 + 5) with 0.9565217391304348
# LAW - Match 1 out of 1 (0 + 1) with 1.0
# LOC - Match 2 out of 2 (1 + 1) with 1.0
# NORP - Match 4 out of 4 (1 + 3) with 1.0
# ORG - Match 312 out of 334 (190 + 122) with 0.9341317365269461
# PERSON - Match 10 out of 10 (7 + 3) with 1.0
# PRODUCT - Match 11 out of 12 (8 + 3) with 0.9166666666666666
# WORK_OF_ART - Match 4 out of 5 (1 + 3) with 0.8
