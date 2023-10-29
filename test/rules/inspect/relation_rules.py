# Date: 2023/03/19 Re-check useful rules before case study
import argparse
import logging
import pickle
import re
from pathlib import Path

import pytest
import spacy
from spacy.matcher import PhraseMatcher

from src import ROOT_DIR
from src.deeplearning.relation.code.tasks.infer import infer_from_trained
from src.utils.typing import RelationPlugin, Span

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_trf")


@pytest.mark.skip(reason="no way of currently testing this")
def test_measure_bert_relation_prec():
    p_all, r_all, f1_all = list(), list(), list()
    preds_list, trues_list = list(), list()
    base_dir = Path(ROOT_DIR) / "pretrained_data/2022_Kfold/relation/"
    args = argparse.Namespace(
        **dict(
            task="istar",
            train_data=str((base_dir / "admin.jsonl").resolve()),
            use_pretrained_blanks=0,
            num_classes=4,
            batch_size=32,
            gradient_acc_steps=1,
            max_norm=1.0,
            fp16=0,
            num_epochs=25,
            lr=7e-05,
            model_no=0,
            model_size="bert-base-uncased",
            train=0,
            infer=1,
        )
    )
    inferer = infer_from_trained(args, detect_entities=False)
    tp, fp, tn, fn = 0, 0, 0, 0
    with open(base_dir / "0/df_test.pkl", "rb") as pkl_file:
        test = pickle.load(pkl_file)
        for index, row in test.iterrows():
            sents = row["sents"]
            relations = row["relations"]
            trues = row["relations_id"]  # no: 1; dependency: 0; isa: 2
            preds = inferer.infer_sentence(sents, detect_entities=False)
            if trues == 1:
                if trues == preds:
                    tn += 1
                else:
                    fp += 1
            elif preds == 1:
                # trues != 1
                fn += 1
            else:
                if trues == preds:
                    tp += 1
                else:
                    fp += 1

        # headers = ["precision", "recall", "f1-score", "support"]
        # head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        # report = head_fmt.format("", *headers, width=5)
        # report += "\n\n"
        # row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        # p = tp / (tp + fp)
        # r = tp / (tp + fn)
        # f1 = 2 * p * r / (p + r)
        # report += row_fmt.format(
        #     *["Relation", r, p, f1, tp + fn], width=5, digits=5
        # )
        # print(report)

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        p_all.append(p)
        r_all.append(r)
        f1_all.append(f1)
        print(p, r, f1, sep="\t")

    K = 1
    print(sum(p_all) / K, sum(r_all) / K, sum(f1_all) / K)
    # print(cr(trues_list, preds_list, digits=8))


def apply_rules(rule: RelationPlugin):
    base_dir = Path(ROOT_DIR) / "pretrained_data/2022_Kfold/relation/"
    p, n = 0, 0
    for i in range(10):
        with open(base_dir / f"{i}/df_test.pkl", "rb") as pkl_file:
            test = pickle.load(pkl_file)
            for index, row in test.iterrows():
                sents = row["sents"]
                relations = row["relations"]
                trues = row[
                    "relations_id"
                ]  # no: 1; dependency: 0; isa: 2; parts-of: 3
                e1_all = re.search(r"\[E1](.*)\[/E1]", sents)
                if e1_all:
                    e1_raw = e1_all.group(1)
                else:
                    e1_raw = None
                e2_all = re.search(r"\[E2](.*)\[/E2]", sents)
                if e2_all:
                    e2_raw = e2_all.group(1)
                else:
                    e2_all = None
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
                if not matches:
                    continue
                e1: Span = min(
                    matches,
                    key=lambda t: abs(t.start - e1_idx),  # type: ignore
                )
                matcher.remove("e1")
                matcher.add("e2", [nlp(e2_raw)])
                matches = matcher(sent, as_spans=True)
                if not matches:
                    continue
                e2: Span = min(
                    matches, key=lambda t: abs(t.start - e2_idx)  # type: ignore
                )

                result = rule(sent, e1, e2)
                if result is not None:
                    if result == trues:
                        p += 1
                    else:
                        print(sents)
                        print(e1, e2)
                        print(trues)
                        n += 1
    print(f"✅: {p}  ❌: {n}")


# TODO: 修改类型
def default(s: Span, e1: Span, e2: Span) -> int | None:
    def find_children(tok, dep=None, pos=None, tag=None, text=None):
        children_list: list[Span] = list()
        if not list(tok.children):
            return children_list
        for t in tok.children:
            flag = True
            if dep and t.dep_ != dep:
                flag = False
            if pos and t.pos != pos:
                flag = False
            if tag and t.tag != tag:
                flag = False
            if text and t.lower_ != text:
                flag = False
            if flag:
                children_list.append(t)
        return children_list

    nsubj = list()
    obj = None

    for token in s:
        if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
            nsubj.append(token)
        if (
            token.dep_ in ["xcomp", "ccomp"] and nsubj
        ):  # and token.head == nsubj.head:
            obj = find_children(token.head, dep="dobj")
            obj.extend(find_children(token, dep="dobj"))
            obj.extend(find_children(token, dep="pobj"))
            obj.extend(find_children(token.head, dep="pobj"))

    if nsubj and obj:
        for n in nsubj:
            for o in obj:
                if n.text in e1.text and o.text in e2.text:
                    return 0
                if n.text in e2.text and o.text in e2.text:
                    return 0

    return None


def agent_pobj(s: Span, e1: Span, e2: Span) -> int | None:
    e1_root = e1.root
    e2_root = e2.root

    if (
        e1_root.dep_ == "pobj"
        and e1_root.head.dep_ == "agent"
        and e2_root.dep_ == "nsubjpass"
        and e1_root.head.head == e2_root.head
    ):
        return 0

    if (
        e2_root.dep_ == "pobj"
        and e2_root.head.dep_ == "agent"
        and e1_root.dep_ == "nsubjpass"
        and e2_root.head.head == e1_root.head
    ):
        return 0

    return None


def conj_exclude(s: Span, e1: Span, e2: Span) -> int | None:
    e1_root = e1.root
    e2_root = e2.root

    if e1_root.head.lower_ == "between" or e2_root.head.lower_ == "between":
        return None

    if e1_root.dep_ == "conj" and e1_root.head == e2_root:
        print(s, e1, e2)
        return 1
    if e2_root.dep_ == "conj" and e2_root.head == e1_root:
        print(s, e1, e2)
        return 1
    return None


def nsubj_attr(s: Span, e1: Span, e2: Span) -> int | None:
    e1_root = e1.root
    e2_root = e2.root
    if e2_root.dep_ == "attr" and e1_root.head == e2_root.head:
        return 2
    if e1_root.dep_ == "attr" and e2_root.head == e1_root.head:
        return 2
    return None


def consists(s: Span, e1: Span, e2: Span) -> int | None:
    e1_root = e1.root
    e2_root = e2.root
    if (
        e1_root.head.lower_ == "consists"
        and e2_root.dep_ == "pobj"
        and e2_root.head.head == e1_root.head
    ):
        return 3
    if (
        e2_root.head.lower_ == "consists"
        and e1_root.dep_ == "pobj"
        and e1_root.head.head == e2_root.head
    ):
        return 3
    return None


def nsubj_pobj(s: Span, e1: Span, e2: Span) -> int | None:
    e1_root = e1.root
    e2_root = e2.root
    nsubj, pobj = None, None
    if e2_root.dep_ == "pobj":
        nsubj = e1_root
        pobj = e2_root
    elif e1_root.dep_ == "pobj":
        nsubj = e2_root
        pobj = e1_root
    else:
        return None

    nsubj_head = nsubj.head
    pobj_org = pobj
    while pobj.has_head():
        if pobj.dep_ == "ROOT":
            break
        if pobj.head == nsubj_head:
            if pobj.dep_ == "dobj" and pobj_org.head.lower_ == "with":
                print(s, e1, e2)
                return 0
            if pobj_org.head.lower_ == "through":
                print(s, e1, e2)
                return 0
        pobj = pobj.head
    return None


def test_rules():
    apply_rules(nsubj_pobj)


if __name__ == "__main__":
    logging.getLogger("src").setLevel(logging.WARNING)
    # test_measure_bert_relation_prec()
