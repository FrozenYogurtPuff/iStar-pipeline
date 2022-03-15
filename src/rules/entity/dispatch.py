from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, List, Optional, Tuple

import spacy_alignments as tokenizations

import src.deeplearning.infer.result as br
from src.rules.utils.seq import is_entity_type_ok
from src.utils.spacy import (
    get_spacy,
    get_token_idx,
    include_elem,
    match_noun_chunk,
)
from src.utils.typing import (
    Alignment,
    BertEntityLabel,
    BertEntityLabelBio,
    EntityFix,
    EntityRulePlugins,
    FixEntityLabel,
    HybridToken,
    SpacySpan,
    is_bert_entity_label,
    is_bert_entity_label_bio_list,
    is_fix_entity_label,
)

from ...deeplearning.infer.utils import label_mapping_bio
from ..config import entity_plugins

logger = logging.getLogger(__name__)

Collector = Tuple[HybridToken, FixEntityLabel]


def collect_filter(
    data: List[Optional[Collector]], sample: Collector
) -> List[Collector]:
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


def simple_bert_merge(
    res: List[EntityFix], b: br.BertResult
) -> List[EntityFix]:
    def simple_check_bert(
        sample: EntityFix, bert: List[BertEntityLabelBio]
    ) -> bool:
        bert_idx = sample.bert_idxes
        label = sample.label
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
    def prob_check_bert(sample: EntityFix) -> Optional[BertEntityLabel]:
        bert_idx = sample.bert_idxes
        label = sample.label

        bert_start, bert_end = bert_idx[0], bert_idx[-1]
        label_m, prob = b.matrix_find_prob_max(bert_start, bert_end)
        assert is_bert_entity_label(label_m)
        mapping_bio = label_mapping_bio(label_m)

        constituous_label = True
        if b.preds[bert_start] != mapping_bio[0]:
            constituous_label = False
        for i in range(bert_start + 1, bert_end + 1):
            if b.preds[i] != mapping_bio[1]:
                constituous_label = False

        if not constituous_label and prob > 0.3:
            return label_m

        if prob < 0.3 and not is_entity_type_ok(label, label_m):
            assert is_bert_entity_label(label)
            return label

        return None

    new_list: List[EntityFix] = list()
    for item in res:
        item_token, item_idx, item_bidx, item_label = item
        # if return label, then need fix
        fix_label = prob_check_bert(item)
        if fix_label:
            assert is_fix_entity_label(fix_label)
            new_list.append(
                EntityFix(item_token, item_idx, item_bidx, fix_label)
            )
    return new_list


# def bert_merge(res: List[EntityFix], b: br.BertResult) -> List[EntityFix]:
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
    bert_func: Callable = prob_bert_merge,
) -> List[EntityFix]:
    collector: List[Optional[Collector]] = list()
    result: List[EntityFix] = list()

    for func in funcs:
        packs = func(s)
        for token, label in set(packs):
            # if `noun_chunk`, use noun chunks instead of tokens
            if noun_chunk:
                c = match_noun_chunk(token, s)
                if c:
                    collector.append((c, label))
                else:
                    collector.append((token, label))
            else:
                collector.append((token, label))

    while collector:
        front = collector.pop()
        assert front is not None
        token, _ = front
        group = collect_filter(collector, front)
        labels = Counter([lab for _, lab in group]).most_common(2)

        # Force not 'Both'
        # if label[0][0] != 'Both' or len(label) == 1:
        #     label = label[0][0]
        # else:
        #     label = label[1][0]
        label = labels[0][0]

        # add_all does not search bert_idx
        idx = get_token_idx(token)
        if add_all:
            result.append(EntityFix(token, idx, [], label))
        else:
            assert s2b is not None
            bert_idx = (
                s2b[idx[0]]
                if len(idx) == 1
                else [s2b[idx[0]][0], s2b[idx[-1]][-1]]
            )
            result.append(EntityFix(token, idx, bert_idx, label))

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
