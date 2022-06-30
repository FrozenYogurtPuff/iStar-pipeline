from __future__ import annotations  # Remove that after Python 3.10

import copy
import dataclasses
import logging
from test.rules.utils.metrics import calc_metrics, log_diff_ents

from src.deeplearning.entity.infer.utils import (
    get_series_bio,
    label_mapping_bio,
    label_mapping_de_bio,
)
from src.utils.typing import BertMatrix, EntityFix, SeqSlicesTuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BertResult:
    preds: list[str]
    trues: list[str] | None
    matrix: BertMatrix
    tokens: list[str]
    labels: list[str]  # labels.txt -like label index

    def matrix_find_prob_max(self, start: int, end: int) -> tuple[str, float]:
        logger.debug(f"Matrix find: {start} - {end}")
        data: dict[str, list[int]] = dict()  # { 'Actor': [1, 2], ... }
        for i, la in enumerate(self.labels):
            if la == "O":
                continue
            key_l, _ = label_mapping_de_bio(la)

            if key_l in data:
                data[key_l].append(i)
            else:
                data[key_l] = [i]

        max_value, max_type = 0.0, None
        for key_d in data:
            temp_value, temp_type = 0.0, key_d
            idx = data[key_d]
            for tok in self.matrix[start : end + 1]:
                for i in idx:
                    temp_value += tok[i]
            if temp_value > max_value:
                max_value, max_type = temp_value, temp_type

        logger.debug(f"Matrix type: {max_type}")
        avg_value = max_value / (end - start + 1)

        assert max_type is not None
        return max_type, avg_value

    def apply_fix(self: BertResult, fixes: list[EntityFix]) -> BertResult:
        new_inst = copy.deepcopy(self)
        for hyb, ali, bali, lab in fixes:
            assert len(bali) in [1, 2]
            if lab == "Both":
                lab_m, avg = self.matrix_find_prob_max(bali[0], bali[-1])
                lab = lab_m

            key = bali[0]
            mapping = label_mapping_bio(lab)
            new_inst.preds[key] = mapping[0]

            if len(bali) == 2 and bali[0] != bali[-1]:
                for i in range(key + 1, bali[-1] + 1):
                    new_inst.preds[i] = mapping[1]

        # TODO: problem searching via low-efficiency ways
        if fixes:
            before_preds, before_trues = get_series_bio([self])
            before_correct = calc_metrics(before_trues, before_preds)
            if self.preds == self.trues:
                logger.info(f"Apply fix - Before: {self}")
            else:
                logger.debug(f"Apply fix - Before: {self}")
            logger.debug(f"Apply fix - Fix: {fixes}")

            after_pred, after_trues = get_series_bio([new_inst])
            after_correct = calc_metrics(after_trues, after_pred)
            if after_correct > before_correct:
                logger.info(f"Apply fix - After: {new_inst}")
            elif after_correct == before_correct:
                logger.debug(f"Apply fix - After: {new_inst}")
            else:
                logger.warning(f"Apply fix - After: {new_inst}")
        else:
            if self.preds == self.trues:
                logger.info(f"Result without fixes: {self}")
            else:
                logger.debug(f"Result without fixes: {self}")

        if fixes:
            pred_entities, true_entities = get_series_bio([new_inst])
            log_diff_ents(true_entities, pred_entities, new_inst)
        else:
            pred_entities, true_entities = get_series_bio([self])
            log_diff_ents(true_entities, pred_entities, self)

        return new_inst

    def apply_slices(
        self: BertResult, slices: list[SeqSlicesTuple]
    ) -> BertResult:
        preds = self.preds[:]
        for slice_ in slices:
            start, end, type_ = slice_
            for i, pred in enumerate(preds[start : end + 1], start=start):
                if pred.endswith("Core") and type_ == "Aux":  # TODO
                    preds[i] = pred.replace("Core", "Aux")
                # if pred.endswith("Aux") and type_ == "Core":
                #     preds[i] = pred.replace("Aux", "Core")
        return BertResult(
            preds, self.trues, self.matrix, self.tokens, self.labels
        )

    def __str__(self):
        return (
            f"{' '.join(self.tokens)}\nTrue: {self.trues}\nPred: {self.preds}"
        )
