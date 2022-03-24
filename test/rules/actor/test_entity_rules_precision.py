import logging
from test.rules.utils.load_dataset import load_dataset

import spacy

from src.rules.config import actor_plugins
from src.rules.entity.dispatch import dispatch
from src.rules.utils.seq import is_entity_type_ok
from src.utils.spacy import char_idx_to_word_idx
from src.utils.typing import DatasetEntityLabel, EntityFix, Span

logger = logging.getLogger(__name__)


# result = [(1, [], 'Actor')]
# labels = [[0, 12, "Actor"], [31, 35, "Actor"], [59, 88, "Resource"]]
def check_result_precision(
    sent: Span, result: list[EntityFix], labels: list[DatasetEntityLabel]
) -> int:
    target = 0
    for item in result:
        _, idx, _, attr = item
        for label in labels:
            begin, end, attr_hat = label
            begin, end = char_idx_to_word_idx(sent, begin, end)
            if begin <= idx[0] <= idx[-1] < end and is_entity_type_ok(
                attr, attr_hat
            ):
                target += 1
    return target


def test_entity_rules_precision():
    nlp: spacy.language.Language = spacy.load("en_core_web_lg")
    res = load_dataset("pretrained_data/entity_ar_r_combined/all.jsonl")
    total = 0
    target = 0
    for i, sent, anno in res:
        logger.debug(f"Before dispatch in test: {sent}")
        s = nlp(sent)
        result = dispatch(s[:], None, None, add_all=True, funcs=actor_plugins)
        cur_length = len(result)
        total += cur_length

        precs = check_result_precision(s[:], result, anno)
        target += precs
        if cur_length != precs:
            logger.warning(f"Sent: {s.text}")
            logger.warning(f"Line {i}: {result}, token: {result[0][0]}")
            logger.warning(f"current Hit {precs} out of {cur_length}")
        else:
            if cur_length != 0:
                logger.info(f"Sent: {s.text}")
                logger.info(f"Line {i}: {result}")
                logger.info(f"current Hit {precs} out of {cur_length}")
            else:
                logger.debug(f"Sent: {s.text}")
                logger.debug(f"Line {i}: {result}")
                logger.debug("current Hit 0 out of 0")
    precision = target / total if total != 0 else 0
    logger.error(f"Total precision: {precision} about {target}/{total}")


if __name__ == "__main__":
    test_entity_rules_precision()
