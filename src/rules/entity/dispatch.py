import logging
from typing import List, Tuple, Union
from collections import Counter

from src.rules.utils.seq import is_entity_type_ok
from src.rules.utils.spacy import get_token_idx, include_elem
from src.utils.typing import (Alignment, BertEntityLabel, EntityFix,
                              EntityRulePlugins, SpacySpan, SpacyToken, FixEntityLabel, HybridToken)

from ..config import entity_plugins, entity_autocrat

logger = logging.getLogger(__name__)

Collector = Tuple[HybridToken, FixEntityLabel, bool]


def collect_filter(data: List[Collector], sample: Collector) -> Tuple[List[Collector], bool]:
    t, l, a = sample
    ret = [sample]
    autocratic = a
    for item in data[:]:
        tok, lab, aut = item
        if tok == t:
            if autocratic and aut:  # Both True, append
                ret.append(item)
            elif autocratic and not aut:  # Already set, pass
                pass
            elif not autocratic and aut:  # Set aut
                ret = [item]
                autocratic = True
            elif not autocratic and not aut:  # Both False, append
                ret.append(item)
            data.remove(item)
    return ret, autocratic


def simple_check_bert(sample: EntityFix, bert: List[BertEntityLabel]) -> bool:
    token, idx, bert_idx, label = sample
    for num in bert_idx:
        if not is_entity_type_ok(label, bert[num]):
            return True
    return False


def dispatch(s: SpacySpan, bert_labels: List[BertEntityLabel],
             s2b: List[Alignment], add_all: bool = False,
             funcs: EntityRulePlugins = entity_plugins,
             autocrat: EntityRulePlugins = (),  # TODO:
             noun_chunk: bool = False) -> List[EntityFix]:

    collector: List[Collector] = list()
    result: List[EntityFix] = list()

    for func in funcs:
        packs = func(s)
        is_autocratic = True if func in entity_autocrat else False
        for token, label in set(packs):
            if noun_chunk:
                for c in list(s.noun_chunks):
                    if include_elem(token, c):
                        collector.append((c, label, is_autocratic))
                        break
            collector.append((token, label, is_autocratic))

    while collector:
        front = collector.pop()
        token, _, _ = front
        group, _ = collect_filter(collector, front)
        label = Counter([lab for _, lab, _ in group]).most_common(1)[0][0]

        idx = get_token_idx(token)
        if add_all:
            bert_idx = None
        else:
            bert_idx = s2b[idx[0]] if len(idx) == 1 else [s2b[idx[0]][0], s2b[idx[-1]][-1]]
        result.append((token, idx, bert_idx, label))

    # TODO: find the fix things due to BERT, fix it
    # Chunk == BERT, good
    # Chunk less than BERT, use BERT
    # Chunk more than BERT, use Chunk
    # Chunk lap with BERT, combine them

    if not add_all:
        result = list(filter(lambda item: simple_check_bert(item, bert_labels), result))

    # unchecked: List[Tuple[SpacyToken, FixEntityLabel]] = list()
    # both = list()
    #
    # logger.debug(f'Before dispatch in dispatch: {s}')
    # for func in funcs:
    #     packs = func(s)
    #
    #     for token, label in set(packs):
    #         if label == 'Both':
    #             both.append(token)
    #         else:
    #             unchecked.append((token, label))
    #
    # # Add both sign 'Both'
    # for token in both:
    #     for token_hat, _ in unchecked:
    #         if token == token_hat:
    #             break
    #     unchecked.append((token, 'Both'))
    #
    # for token, label in unchecked:
    #     idx = get_token_idx(token)
    #     if add_all:
    #         logger.debug(f"{s.text}\ntoken: {token.text}, idx: {idx}, label: {label}")
    #         result.append((token, idx, [], label))
    #     else:
    #         select = False
    #         map_idx = s2b[idx]
    #         for num in map_idx:
    #             if not is_entity_type_ok(label, bert_labels[num]):
    #                 select = True
    #         if select:
    #             logger.debug(f"{s.text}\ntoken: {token.text}, idx: {idx},"
    #                          f"map_idx:{map_idx}, label: {label}")
    #             result.append((token, idx, map_idx, label))

    return result
