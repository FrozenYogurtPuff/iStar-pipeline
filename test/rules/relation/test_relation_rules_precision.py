import logging
import pickle
import re
from test.rules.inspect.relation_rules import (
    agent_pobj,
    conj_exclude,
    consists,
    nsubj_attr,
    nsubj_pobj,
)

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

logger = logging.getLogger(__name__)


def test_relation_rules_precision():
    relation_plugins_new = (
        conj_exclude,
        nsubj_attr,
        consists,
        agent_pobj,
        nsubj_pobj,
    )
    nlp: spacy.language.Language = spacy.load("en_core_web_trf")
    tp_fp, tp_fn, tp = 0, 0, 0
    for i in range(10):
        with (
            open(
                f"pretrained_data/2022_Kfold/relation/{i}/df_test.pkl", "rb"
            ) as pkl_file
        ):
            test = pickle.load(pkl_file)
            for index, row in test.iterrows():
                sents = row["sents"]
                relations = row["relations"]
                trues = row["relations_id"]  # no: 1; dependency: 0; isa: 2

                e1_all = re.search(r"\[E1](.*)\[/E1]", sents)
                if e1_all:
                    e1_raw = e1_all.group(1)
                else:
                    e1_raw = None
                e2_all = re.search(r"\[E2](.*)\[/E2]", sents)
                if e2_all:
                    e2_raw = e2_all.group(1)
                else:
                    e2_raw = None
                e1_idx, e2_idx = None, None
                for idx, tok in enumerate(sents.split(" ")):
                    if "[E1]" in tok:
                        e1_idx = idx
                    if "[E2]" in tok:
                        e2_idx = idx
                assert e1_idx is not None
                assert e2_idx is not None
                sent = nlp(
                    sents.replace("[E1]", "")
                    .replace("[/E1]", "")
                    .replace("[E2]", "")
                    .replace("[/E2]", "")
                )[:]
                matcher = PhraseMatcher(nlp.vocab, attr="ORTH")
                matcher.add("e1", [nlp(e1_raw)])
                matches = matcher(sent, as_spans=True)
                check = True
                if not matches:
                    check = False
                else:
                    e1: Span = min(
                        matches,
                        key=lambda t: abs(  # type: ignore
                            t.start - e1_idx  # type: ignore
                        ),
                    )
                matcher.remove("e1")
                matcher.add("e2", [nlp(e2_raw)])
                matches = matcher(sent, as_spans=True)
                if not matches:
                    check = False
                else:
                    e2: Span = min(
                        matches,
                        key=lambda t: abs(  # type: ignore
                            t.start - e2_idx  # type: ignore
                        ),
                    )

                if check:
                    for func in relation_plugins_new:
                        result = func(sent, e1, e2)
                        if result:
                            if result != 1:
                                tp_fp += 1
                            if trues != 1:
                                tp_fn += 1
                            if result == trues and trues != 1:
                                tp += 1

    precision = tp / tp_fp if tp_fp != 0 else 0
    recall = tp / tp_fn if tp_fn != 0 else 0
    logger.error(f"Total precision: {precision} about {tp}/{tp_fp}")
    logger.error(f"Total recall: {recall} about {tp}/{tp_fn}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    test_relation_rules_precision()
