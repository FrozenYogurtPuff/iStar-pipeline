from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.deeplearning.entity.infer.result import BertResult
from src.deeplearning.entity.utils.utils_metrics import get_entities_bio

logger = logging.getLogger(__name__)


class LabelTypeException(Exception):
    pass


def get_series_bio(src: list[BertResult], func: Callable = get_entities_bio):
    p = [s.preds for s in src]
    t = [s.trues for s in src]
    return func(p), func(t)


def get_list_bio(src: list[BertResult]):
    p = list()
    t = list()

    for item in src:
        if item.trues:
            for token in item.trues:
                t.append(token)
        for token in item.preds:
            p.append(token)

    return p, t


def label_mapping_bio(lab: str) -> tuple[str, str]:
    if lab in ["Role", "Agent", "Resource", "Actor", "Core", "Cond", "Aux"]:
        return f"B-{lab}", f"I-{lab}"
    if lab in ["O"]:
        return "O", "O"
    if lab == "Quality":
        logger.error("Label Quality")
        # I think it should not be occurred
    logger.error(f"Illegal label {lab}")
    raise LabelTypeException("Unexpected label type")


def label_mapping_de_bio(lab: str) -> tuple[str, bool]:
    def label_check() -> str:
        if not lab.startswith("B-") and not lab.startswith("I-"):
            logger.error(f"Illegal label-bio {lab}")
            raise LabelTypeException("Unexpected label-bio type")
        if lab.endswith(
            ("Role", "Agent", "Actor", "Resource", "Core", "Cond", "Aux")
        ):
            return lab[2:]

        logger.error(f"Illegal label-bio {lab}")
        raise LabelTypeException("Unexpected label-bio type")

    flag = True if lab.startswith("B-") else False
    return label_check(), flag
