import json
import spacy
import logging
from typing import List
from pathlib import Path

from src import ROOT_DIR
from src.rules.config import entity_plugins
from src.rules.entity.dispatch import dispatch
from src.rules.utils.seq import is_entity_type_ok
from src.rules.utils.spacy import char_idx_to_word_idx
from src.typing import DatasetLabel, EntityFix, SpacySpan

logger = logging.getLogger(__name__)


def load_dataset():
    with open(Path(ROOT_DIR).joinpath('pretrained_data/entity_ar_r_combined/all.jsonl'), 'r') as j:
        for idx, line in enumerate(j):
            a = json.loads(line)
            sent = a['text']
            anno = a['labels']
            yield idx, sent, anno


# result = [(1, [], 'Actor')]
# labels = [[0, 12, "Actor"], [31, 35, "Actor"], [59, 88, "Resource"]]
def check_result_precision(sent: SpacySpan, result: List[EntityFix], labels: List[DatasetLabel]) -> int:
    target = 0
    for item in result:
        _, idx, _, attr = item
        for label in labels:
            begin, end, attr_hat = label
            begin, end = char_idx_to_word_idx(sent, begin, end)
            if begin <= idx < end and is_entity_type_ok(attr, attr_hat):
                target += 1
                break
    return target


def test_entity_rules_precision():
    nlp: spacy.language.Language = spacy.load('en_core_web_lg')
    res = load_dataset()
    total = 0
    target = 0
    for i, sent, anno in res:
        s = nlp(sent)
        result = dispatch(s[:], [], [], add_all=True, funcs=[entity_plugins[2]])
        cur_length = len(result)
        total += cur_length

        precs = check_result_precision(s[:], result, anno)
        target += precs
        if cur_length != precs:
            logger.warning(f'Sent: {s.text}')
            logger.warning(f'Line {i}: {result}, token: {result[0][0]}')
            logger.warning(f'current Hit {precs} out of {cur_length}')
        else:
            logger.debug(f'Line {i}: {result}')
            logger.debug(f'current Hit {precs} out of {cur_length}')
    precision = target / total
    logger.error(f'Total precision: {precision} about {target}/{total}')


if __name__ == '__main__':
    test_entity_rules_precision()