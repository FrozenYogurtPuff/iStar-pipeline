from __future__ import annotations

import logging
from collections import Counter
from typing import Callable

from spacy.tokens import Span, Token

from src.deeplearning.entity.infer.result import BertResult
from src.deeplearning.entity.infer.utils import (
    get_series_bio,
    label_mapping_bio,
)
from src.rules.config import actor_plugins
from src.rules.utils.seq import get_s2b_idx, is_entity_type_ok
from src.utils.spacy_utils import get_token_idx, include_elem, match_noun_chunk
from src.utils.typing import Alignment, EntityFix, RulePlugins

logger = logging.getLogger(__name__)

Collector = tuple[Token | Span, str]


def collect_filter(
    data: list[Collector | None], sample: Collector
) -> list[Collector]:
    t, lab = sample
    ret = [sample]
    for idx, item in enumerate(data[:]):
        assert item is not None
        tok, lab = item
        if include_elem(tok, t):
            ret.append(item)
            data[idx] = None
    while None in data:
        data.remove(None)
    return ret


def simple_bert_merge(res: list[EntityFix], b: BertResult) -> list[EntityFix]:
    def simple_check_bert(sample: EntityFix, bert: list[str]) -> bool:
        bert_idx = sample.bert_idxes
        label = sample.label
        for num in bert_idx:
            if not is_entity_type_ok(label, bert[num]):
                return True
        return False

    preds = b.preds

    new_list: list[EntityFix] = list()
    for item in res:
        if not simple_check_bert(item, preds):
            continue
        new_list.append(item)
    return new_list


def prob_bert_merge(res: list[EntityFix], b: BertResult) -> list[EntityFix]:
    def prob_check_bert(sample: EntityFix) -> str | None:
        bert_idx = sample.bert_idxes
        label = sample.label

        bert_start, bert_end = bert_idx[0], bert_idx[-1]
        label_m, prob = b.matrix_find_prob_max(bert_start, bert_end)
        mapping_bio = label_mapping_bio(label_m)

        # 标签是否连续，即除去第一个 B-X 其它是否为 I-X
        constituous_label = True
        if b.preds[bert_start] not in mapping_bio:
            constituous_label = False
        for i in range(bert_start + 1, bert_end + 1):
            if b.preds[i] != mapping_bio[1]:
                constituous_label = False

        logger.debug(f"label: {label}, label_m: {label_m}")
        logger.debug(
            f"constituous: {constituous_label}, prob: {prob}, type_ok: {is_entity_type_ok(label, label_m)}"
        )

        # 若标签不连续但概率高则保留 BERT 意见
        if not constituous_label and prob > 0.8:
            return label_m

        # 当 BERT 非 O 标签与 规则不同，且非 O 标签概率低时选择规则
        if prob < 0.4 and not is_entity_type_ok(label, label_m):
            return label

        return None

    new_list: list[EntityFix] = list()
    for item in res:
        item_token, item_idx, item_bidx, item_label = item
        # if return label, then need fix
        fix_label = prob_check_bert(item)
        if fix_label:
            new_list.append(
                EntityFix(item_token, item_idx, item_bidx, fix_label)
            )
    return new_list


# Filter: Result -> Partial Result
def exclude_actor(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
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


def exclude_intention_verb(
    sp: Span, b: BertResult | None, s2b: list[Alignment] | None
) -> list[EntityFix]:
    exclude_list = list()
    assert b is not None
    pred_entities, _ = get_series_bio([b])
    for label, s, e in pred_entities:
        # if sp[e].tag_.startswith("JJ"):
        #     if e - 1 > 0 and sp[e - 1].lower_ not in ["am", "is", "are", "be", "was", "were"]:
        #         bert_idx = get_s2b_idx(s2b, [s, e])
        #         exclude_list.append(EntityFix(sp[s:e + 1], [s, e], bert_idx, "O"))
        if not sp[e].tag_.startswith("VB") and not sp[e].tag_.startswith("NN"):
            assert s2b is not None
            bert_idx = get_s2b_idx(s2b, [s, e])
            exclude_list.append(
                EntityFix(sp[s : e + 1], [s, e], bert_idx, "O")
            )
    return exclude_list


def dispatch(
    s: Span,
    b: BertResult | None,
    s2b: list[Alignment] | None,
    add_all: bool = False,
    noun_chunk: bool = False,
    funcs: RulePlugins = actor_plugins,
    bert_func: Callable = prob_bert_merge,
) -> list[EntityFix]:
    collector: list[Collector | None] = list()
    result: list[EntityFix] = list()

    INCLUDE = True
    EXCLUDE = True

    if INCLUDE:
        for func in funcs:
            packs = func(s)
            for token, label in set(packs):
                # if `noun_chunk`, use noun chunks instead of tokens
                if noun_chunk:
                    c = match_noun_chunk(token, s)
                    if c:
                        logger.debug(f"Noun chunk hit: {c}")
                        collector.append((c, label))
                    else:
                        collector.append((token, label))
                else:
                    collector.append((token, label))

    logger.info(collector)
    logger.debug(s2b)

    while collector:
        front = collector.pop()
        assert front is not None
        token, _ = front
        group = collect_filter(collector, front)
        labels = Counter([lab for _, lab in group]).most_common(2)

        label = labels[0][0]

        # add_all does not search bert_idx
        idx = get_token_idx(token)
        if add_all:
            result.append(EntityFix(token, idx, [], label))
        else:
            assert s2b is not None

            bert_idx = get_s2b_idx(s2b, idx)
            logger.debug(idx)
            logger.debug(bert_idx)
            if not bert_idx:
                continue
            result.append(EntityFix(token, idx, bert_idx, label))

    # hybrid bert to filter consequences
    # `add_all` pass the raw result without considering bert
    if not add_all:
        result = bert_func(result, b)

    if EXCLUDE:
        result.extend(exclude_intention_verb(s, b, s2b))

    return result
