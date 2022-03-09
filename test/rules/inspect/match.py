import logging
from test.rules.utils.load_dataset import load_dataset
from typing import Dict

from src.utils.spacy import get_spacy
from src.utils.typing import FixIntentionLabel, SpacySpan

logger = logging.getLogger(__name__)

nlp_dict: Dict[str, SpacySpan] = dict()


def if_inside(sentence: SpacySpan, dep: str = "xcomp") -> bool:
    for token in sentence:
        if token.dep_ == dep:
            return True
    return False


def cache_nlp(s: str) -> SpacySpan:
    global nlp_dict
    try:
        res = nlp_dict[s]
    except KeyError:
        res = nlp(s)[:]
        nlp_dict[s] = res
    return res


if __name__ == "__main__":
    data = list(load_dataset("pretrained_data/task_core_aux_cond/all.jsonl"))

    nlp = get_spacy()
    aux: FixIntentionLabel = "Aux"

    check_list = [
        "acl",
        "acomp",
        "advcl",
        "advmod",
        "agent",
        "amod",
        "appos",
        "attr",
        "aux",
        "auxpass",
        "case",
        "cc",
        "ccomp",
        "compound",
        "conj",
        "csubj",
        "csubjpass",
        "dative",
        "dep",
        "det",
        "dobj",
        "expl",
        "intj",
        "mark",
        "meta",
        "neg",
        "nmod",
        "npadvmod",
        "nsubj",
        "nsubjpass",
        "nummod",
        "oprd",
        "parataxis",
        "pcomp",
        "pobj",
        "poss",
        "preconj",
        "predet",
        "prep",
        "prt",
        "punct",
        "quantmod",
        "relcl",
        "xcomp",
    ]

    for check in check_list:
        total, match = 0, 0
        for i, sent, anno in data:
            sent_processed = cache_nlp(sent)
            if if_inside(sent_processed, check):
                total += 1
                for start, end, lab in anno:
                    if lab == aux:
                        match += 1
                        break

        if total > 0 and match / total > 0.4:
            logger.error(
                f"{check} - Match {match} out of {total} with {match / total if total else 0}"
            )
        else:
            logger.warning(
                f"{check} - Match {match} out of {total} with {match / total if total else 0}"
            )

# agent - Match 41 out of 55 with 0.7454545454545455
# oprd - Match 5 out of 9 with 0.5555555555555556
# relcl - Match 130 out of 187 with 0.6951871657754011
