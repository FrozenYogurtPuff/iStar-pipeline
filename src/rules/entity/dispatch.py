from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, List, Optional, Tuple

import spacy_alignments as tokenizations

import src.deeplearning.infer.result as br
from src.rules.utils.seq import is_entity_type_ok
from src.utils.spacy import get_spacy, get_token_idx, include_elem
from src.utils.typing import (
    Alignment,
    BertEntityLabelBio,
    EntityFix,
    EntityRulePlugins,
    FixEntityLabel,
    HybridToken,
    SpacySpan,
    is_bert_entity_label_bio_list,
    is_fix_entity_label,
)

from ..config import entity_autocrat, entity_plugins

logger = logging.getLogger(__name__)

Collector = Tuple[HybridToken, FixEntityLabel, bool]


def collect_filter(
    data: List[Optional[Collector]], sample: Collector
) -> Tuple[List[Collector], bool]:
    t, l, autocratic = sample
    ret = [sample]
    for idx, item in enumerate(data[:]):
        assert item is not None
        tok, lab, aut = item
        if include_elem(tok, t):
            if autocratic and aut:  # Both True, append
                ret.append(item)
            elif autocratic and not aut:  # Already set, pass
                pass
            elif not autocratic and aut:  # Set aut
                ret = [item]
                autocratic = True
            elif not autocratic and not aut:  # Both False, append
                ret.append(item)
            data[idx] = None
    while None in data:
        data.remove(None)
    return ret, autocratic


def simple_bert_merge(
    res: List[EntityFix], b: br.BertResult
) -> List[EntityFix]:
    def simple_check_bert(
        sample: EntityFix, bert: List[BertEntityLabelBio]
    ) -> bool:
        token, idx, bert_idx, label = sample
        for num in bert_idx:
            if not is_entity_type_ok(label, bert[num]):
                return True
        return False

    preds = b.preds
    assert is_bert_entity_label_bio_list(preds)

    new_list: List[EntityFix] = list()
    for item in res:
        if not simple_check_bert(item, preds):
            continue
        new_list.append(item)
    return new_list


# prob soft, filter 'Both' and prob < 0.5
def prob_bert_merge(res: List[EntityFix], b: br.BertResult) -> List[EntityFix]:
    def prob_check_bert(
        sample: EntityFix, bert: List[BertEntityLabelBio]
    ) -> bool:
        token, idx, bert_idx, label = sample
        label_m, prob = b.matrix_find_prob_max(bert_idx[0], bert_idx[-1])
        assert is_fix_entity_label(label_m)
        label = label_m
        if prob < 0.5:
            return False
        for num in bert_idx:
            if not is_entity_type_ok(label, bert[num]):
                return True
        return False

    preds = b.preds
    assert is_bert_entity_label_bio_list(preds)

    new_list: List[EntityFix] = list()
    for item in res:
        if not prob_check_bert(item, preds):
            continue
        new_list.append(item)
    return new_list


# def bert_merge(res: List[EntityFix], b: BertResult) -> List[EntityFix]:
#     # Chunk == BERT, good
#     # Chunk less than BERT, use BERT
#     # Chunk more than BERT, use Chunk
#     # Chunk lap with BERT, combine them or treat differently TODO
#     preds = b.preds
#     # 怎么判断？
#     # 对token指示的上下文进行迭代，匹配概率大小 TODO: need a threshold prob?
#     for token, idx, bert_idx, label in res:


def dispatch(
    s: SpacySpan,
    b: Optional[br.BertResult],
    s2b: Optional[List[Alignment]],
    add_all: bool = False,
    noun_chunk: bool = True,
    funcs: EntityRulePlugins = entity_plugins,
    autocrat: EntityRulePlugins = entity_autocrat,
    bert_func: Callable = prob_bert_merge,
) -> List[EntityFix]:
    collector: List[Optional[Collector]] = list()
    result: List[EntityFix] = list()

    for func in funcs:
        packs = func(s)
        is_autocratic = True if func in autocrat else False
        for token, label in set(packs):
            find = False
            # if `noun_chunk`, use noun chunks instead of tokens
            if noun_chunk:
                for c in list(s.noun_chunks):
                    if include_elem(token, c):
                        collector.append((c, label, is_autocratic))
                        find = True
                        break
            if not find:
                collector.append((token, label, is_autocratic))

    while collector:
        front = collector.pop()
        assert front is not None
        token, _, _ = front
        group, _ = collect_filter(collector, front)
        labels = Counter([lab for _, lab, _ in group]).most_common(2)

        # Force not 'Both'
        # if label[0][0] != 'Both' or len(label) == 1:
        #     label = label[0][0]
        # else:
        #     label = label[1][0]
        label = labels[0][0]

        # add_all does not search bert_idx
        idx = get_token_idx(token)
        if add_all:
            result.append((token, idx, [], label))
        else:
            assert s2b is not None
            bert_idx = (
                s2b[idx[0]]
                if len(idx) == 1
                else [s2b[idx[0]][0], s2b[idx[-1]][-1]]
            )
            result.append((token, idx, bert_idx, label))

    # hybrid bert to filter consequences
    if not add_all:
        result = bert_func(result, b)

    return result


def get_rule_fixes(sent: str, b: br.BertResult) -> List[EntityFix]:
    nlp = get_spacy()
    s = nlp(sent)[:]
    spacy_tokens = [i.text for i in s]
    s2b, _ = tokenizations.get_alignments(spacy_tokens, b.tokens)
    result = dispatch(s, b, s2b)
    return result
