import logging
from test.rules.inspect.utils import cache_nlp, ner_list
from test.rules.utils.load_dataset import load_dataset

from src.utils.spacy import char_idx_to_word, get_spacy, include_elem
from src.utils.typing import BertEntityLabel

logger = logging.getLogger(__name__)


class LabelTypeException(Exception):
    pass


if __name__ == "__main__":
    data = list(load_dataset("pretrained_data/entity_ar_r_combined/all.jsonl"))

    nlp = get_spacy()

    actor: BertEntityLabel = "Actor"
    resource: BertEntityLabel = "Resource"

    # for check in dep_list:
    # for check in tag_list:
    for check in ner_list:
        total, match = 0, 0
        match_resource, match_actor = 0, 0
        for i, sent, anno in data:
            sent_processed = cache_nlp(nlp, sent)
            # for token in sent_processed:
            for token in sent_processed.ents:
                # if token.dep_ == check:
                if token.label_ == check:
                    total += 1
                    for start, end, lab in anno:
                        ground = char_idx_to_word(sent_processed, start, end)
                        if include_elem(token, ground):
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
