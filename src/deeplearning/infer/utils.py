from __future__ import annotations

import logging
from typing import Callable, List, Tuple

import src.deeplearning.infer.result as br
from src.deeplearning.utils.utils_metrics import get_entities
from src.utils.typing import (
    BertEntityLabelBio,
    BertIntentionLabelBio,
    BertUnionLabel,
    BertUnionLabelBio,
)

logger = logging.getLogger(__name__)

actor: Tuple[BertEntityLabelBio, BertEntityLabelBio] = ("B-Actor", "I-Actor")
resource: Tuple[BertEntityLabelBio, BertEntityLabelBio] = (
    "B-Resource",
    "I-Resource",
)
core: Tuple[BertIntentionLabelBio, BertIntentionLabelBio] = (
    "B-Core",
    "I-Core",
)
cond: Tuple[BertIntentionLabelBio, BertIntentionLabelBio] = (
    "B-Cond",
    "I-Cond",
)
aux: Tuple[BertIntentionLabelBio, BertIntentionLabelBio] = ("B-Aux", "I-Aux")


class LabelTypeException(Exception):
    pass


def get_series_bio(src: List[br.BertResult], func: Callable = get_entities):
    p = [s.preds for s in src]
    t = [s.trues for s in src]
    return func(p), func(t)


def label_mapping_bio(
    lab: BertUnionLabel,
) -> Tuple[BertUnionLabelBio, BertUnionLabelBio]:
    if lab == "Actor":
        return actor
    if lab == "Resource":
        return resource
    if lab == "Core":
        return core
    if lab == "Cond":
        return cond
    if lab == "Aux":
        return aux
    if lab == "Quality":
        logger.error("Label Quality")
        # I think it should not be occurred
    logger.error(f"Illegal label {lab}")
    raise LabelTypeException("Unexpected label type")


def label_mapping_de_bio(
    lab: BertUnionLabelBio,
) -> Tuple[BertUnionLabel, bool]:
    def label_check() -> BertUnionLabel:
        if lab in actor:
            return "Actor"
        if lab in resource:
            return "Resource"
        if lab in core:
            return "Core"
        if lab in cond:
            return "Cond"
        if lab in aux:
            return "Aux"
        logger.error(f"Illegal label-bio {lab}")
        raise LabelTypeException("Unexpected label-bio type")

    flag = True if lab.startswith("B-") else False
    return label_check(), flag
