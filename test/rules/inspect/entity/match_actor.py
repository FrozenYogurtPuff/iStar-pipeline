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
    data = list(load_dataset("pretrained_data/2022/actor/divided/all.jsonl"))

    nlp = get_spacy()

    agent = "Agent"
    role = "Role"

    mode: Literal["dep", "tag", "ner"] = "dep"
    check_list = eval(f"{mode}_list")
    include: Literal["", "conj", "head"] = "head"

    for check in check_list:
        # check head
        total_list = list()
        agent_list = list()
        role_list = list()

        total, match = 0, 0
        match_agent, match_role = 0, 0
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
                                if lab == agent:
                                    match_agent += 1
                                    agent_list.append(token.head.dep_)
                                elif lab == role:
                                    match_role += 1
                                    role_list.append(token.head.dep_)
                                else:
                                    raise LabelTypeException(
                                        f"Illegal label type {lab}"
                                    )
                                break

        if total > 0 and match / total > 0.4:
            logger.error(
                f"{check} - Match {match} out of {total} ({match_role} + {match_agent}) with {match / total if total else 0}"
            )
        else:
            logger.warning(
                f"{check} - Match {match} out of {total} ({match_role} + {match_agent}) with {match / total if total else 0}"
            )

        total_counter = Counter(total_list)
        agent_counter = Counter(agent_list)
        role_counter = Counter(role_list)
        for dep_key in dep_list:
            if dep_key in total_counter:
                target = agent_counter[dep_key] + role_counter[dep_key]
                prob = target / total_counter[dep_key]
                if prob > 0.8 and target > 3:
                    logger.warning(
                        f"> {dep_key} - {prob} - {role_counter[dep_key]}+{agent_counter[dep_key]}/{total_counter[dep_key]}"
                    )

# ner
# GPE - Match 19 out of 23 (2 + 17) with 0.8260869565217391
# ORG - Match 242 out of 334 (75 + 167) with 0.7245508982035929
# PERSON - Match 7 out of 10 (0 + 7) with 0.7

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
