from __future__ import annotations

import logging

from spacy.tokens import Span, Token

from src.deeplearning.entity.infer.result import BertResult
from src.deeplearning.entity.infer.utils import get_series_bio
from src.rules.utils.seq import get_s2b_idx
from src.utils.spacy_utils import get_token_idx
from src.utils.typing import Alignment, EntityFix, RulePlugins

logger = logging.getLogger(__name__)

Collector = tuple[Token | Span, str]

# Filter: Result -> Partial Result
def exclude_actor(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].pos_ not in ["NOUN", "PROPN"]:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


def no_pron(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].pos_ == "PRON":
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


def no_parentheses(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        pre, post = False, False
        for i in range(0, s):
            if "(" in sp[i].lower_:
                pre = True
            if ")" in sp[i].lower_:
                pre = False
        for i in range(len(sp) - 1, s, -1):
            if ")" in sp[i].lower_:
                post = True
            if "(" in sp[i].lower_:
                post = False
        if pre and post:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            print(sp[s : e + 1])
            print(sp.sent)
    return exclude_list


def no_such_as(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        head = sp[e]
        such_as = False
        while head.dep_ != "ROOT" and head.has_head():
            if (
                head.lower_ == "as"
                and head.i > 0
                and head.nbor(-1).lower_ == "such"
            ):
                such_as = True
            head = head.head
        if such_as:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


def no_ie(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        head = sp[e]
        ie = False
        while head.dep_ != "ROOT" and head.has_head():
            for child in head.children:
                if child.lower_ == "i.e.":
                    ie = True
            head = head.head
        if ie:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


def no_allow(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].lower_.startswith("allow"):
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


# E(I) for Actor version
def exclude_intention_verb_for_actor(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        # if label == "Role":
        #     continue
        if not sp[e].tag_.startswith("VB") and not sp[e].tag_.startswith("NN"):
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            print(sp[s : e + 1])
            print(sp.sent)
    return exclude_list


# E(I)* version
def exclude_intention_verb(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        # if sp[e].tag_.startswith("JJ"):
        #     if sp[e].has_head() and sp[e].head.lower_ in [
        #         "am",
        #         "is",
        #         "are",
        #         "be",
        #         "was",
        #         "were",
        #     ]:
        #         pass
        #     elif e - 1 > 0 and sp[e - 1].lower_ in [
        #         "am",
        #         "is",
        #         "are",
        #         "be",
        #         "was",
        #         "were",
        #     ]:
        #         pass
        #     else:
        #         print("去掉", sp[e].head, sp[s : e + 1])
        #         assert s2b is not None
        #         bert_idx = get_s2b_idx(s2b, [s, e])
        #         exclude_list.append(
        #             EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
        #         )
        #         ...
        if (
            not sp[e].tag_.startswith("VB")
            and not sp[e].tag_.startswith("NN")
            and not sp[e].tag_.startswith("JJ")
        ):
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            print(sp[s : e + 1])
            print(sp.sent)

    return exclude_list


def exclude_single_det(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].pos_ == "DET":
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


def exclude_trailing_stuff(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].lower_ in ["to"]:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            print(sp[s : e + 1])
            print(sp.sent)
        elif sp[e].dep_ in ["cc", "prep"]:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            new_bert_idx = get_s2b_idx(s2b, [s, e - 1])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            exclude_list.append(
                EntityFix(sp[s:e], [s, e - 1], new_bert_idx, label)
            )
            print(sp[s : e + 1])
            print(sp.sent)
    return exclude_list


def exclude_single_pron(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].pos_ == "PRON":
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            print(sp[s : e + 1])
            print(sp.sent)
    return exclude_list


def all_propn_agent(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        flag = True
        for i in range(s, e + 1):
            if sp[e].pos_ != "PROPN":
                flag = False
        if flag:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "Agent")
            )
            print(sp[s : e + 1])
            print(sp.sent)
    return exclude_list


def correct_be(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if s > 0 and sp[s - 1].lower_ == "be" and sp[s].pos_ == "ADJ":
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            new_bert_idx = get_s2b_idx(s2b, [s - 1, s - 1])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            exclude_list.append(
                EntityFix(sp[s - 1 : e], [s - 1, s - 1], new_bert_idx, "Core")
            )
    return exclude_list


def correct_allow(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].lower_ == "allow" and sp[e + 1].pos_.startswith("V"):
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            new_bert_idx = get_s2b_idx(s2b, [e + 1, e + 1])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            exclude_list.append(
                EntityFix(
                    sp[e + 1 : e + 2], [e + 1, e + 1], new_bert_idx, "Core"
                )
            )
    return exclude_list


def be_able(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].pos_ == "AUX" and sp[e + 1].lower_ in ["able", "required"]:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


def no_aux(
    sp: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if sp[e].pos_ == "AUX" and sp[e].lower_ != "be":
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


def after_neg(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        if s > 0 and sp[s - 1].lower_ in ["without"]:
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
            print(sp[s : e + 1])
            print(sp.sent)
    return exclude_list


def xcomp_ask(
    s: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    both: str = "Both"
    result = list()

    for token in s:
        if token.dep_ == "xcomp":
            head = token.head
            for child in head.children:
                if child.dep_ == "dobj":
                    idx = get_token_idx(child)
                    bert_idx = get_s2b_idx(s2b, idx)
                    result.append(EntityFix(child, idx, bert_idx, "Both"))

    return result


def tag_base(
    s: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    def tag_label(tag: str, head_dep: str) -> str | None:
        both: str = "Both"

        if (tag, head_dep) in [
            #     ("DT", "nsubj"),
            ("HYPH", "nsubj"),
            ("NNP", "nsubj"),
        ]:
            return both
        elif (tag, head_dep) in [
            #     ("NNP", "acl"),
            ("NNS", "poss"),
        ]:
            return both
        elif (tag, head_dep) in [("POS", "nsubj")]:
            return both

        return None

    def tag_head_label(tag: str, head_dep: str) -> str | None:
        both: str = "Both"
        if (tag, head_dep) in [
            #     ("DT", "nsubj"),
            ("HYPH", "nsubj"),
            #     ("NNP", "nsubj"),
        ]:
            return both
        elif (tag, head_dep) in [
            ("NNS", "poss"),
        ]:
            return both
        elif (tag, head_dep) in [
            ("POS", "nsubj"),
            #     ("VBZ", "appos"),
        ]:
            return both

        return None

    result = list()

    for token in s:
        tag = token.tag_
        head_dep = token.head.dep_
        label = tag_label(tag, head_dep)
        if label:
            idx = get_token_idx(token)
            bert_idx = get_s2b_idx(s2b, idx)
            result.append(EntityFix(token, idx, bert_idx, "Both"))
            print(token)
            print(token.sent)
        head_label = tag_head_label(tag, head_dep)
        if head_label:
            idx = get_token_idx(token.head)
            bert_idx = get_s2b_idx(s2b, idx)
            result.append(EntityFix(token.head, idx, bert_idx, "Both"))
            print(token)
            print(token.sent)
    return result


def acl_to(
    s: Span, b: BertResult | None, s2b: list[Alignment]
) -> list[EntityFix]:
    core: str = "Core"
    result = list()

    for token in s:
        if token.dep_ == "acl":
            for child in token.children:
                if child.dep_ == "aux" and child.lower_ == "to":
                    idx = get_token_idx(token)
                    bert_idx = get_s2b_idx(s2b, idx)
                    result.append(EntityFix(token, idx, bert_idx, "Core"))

    return result


funcs_ae = (
    exclude_intention_verb_for_actor,
    no_parentheses,
    exclude_single_det,
    exclude_trailing_stuff,
    exclude_single_pron,
    # tag_base,
)

funcs_ie = (
    exclude_intention_verb,
    no_parentheses,
    after_neg,
)


def dispatch(
    s: Span,
    b: BertResult | None,
    s2b: list[Alignment] | None,
    funcs: RulePlugins = funcs_ae,
) -> list[EntityFix]:
    result: list[EntityFix] = list()

    for func in funcs:
        res = func(s, b, s2b)
        result.extend(res)

    return result
