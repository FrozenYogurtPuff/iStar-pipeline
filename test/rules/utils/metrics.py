import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.deeplearning.entity.infer.result import BertResult

logger = logging.getLogger(__name__)


def calc_metrics(true_entities, pred_entities) -> float:
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    logger.debug(d2)

    total_correct, total_pred, total_true = 0, 0, 0

    for type_name, type_true_entities in d1.items():
        type_pred_entities = d2[type_name]
        nb_correct = len(type_true_entities & type_pred_entities)
        nb_pred = len(type_pred_entities)
        nb_true = len(type_true_entities)

        total_correct += nb_correct
        total_pred += nb_pred
        total_true += nb_true

    if total_true == 0:
        return 0
    return total_correct / total_true


def log_diff_ents(true_entities, pred_entities, res: "BertResult"):
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    for type_name, type_true_entities in d1.items():
        type_pred_entities = d2[type_name]
        nb_diff = type_true_entities - (
            type_true_entities & type_pred_entities
        )
        for diff in nb_diff:
            start, end = diff
            logger.warning(
                f"[{type_name}] Should have {res.tokens[start:end+1]}"
            )
        nb_diff = type_pred_entities - (
            type_true_entities & type_pred_entities
        )
        for diff in nb_diff:
            start, end = diff
            logger.warning(
                f"[{type_name}] Should not have {res.tokens[start:end + 1]}"
            )
