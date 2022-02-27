from __future__ import annotations  # Remove that after Python 3.10

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

from src.deeplearning.infer.utils import (
    label_mapping_bio,
    label_mapping_de_bio,
)
from src.utils.typing import (
    BertMatrix,
    BertUnionLabel,
    BertUnionLabelBio,
    EntityFix,
    Token,
    is_bert_union_label,
    is_fix_entity_label,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BertResult:
    preds: List[BertUnionLabelBio]
    trues: Optional[List[BertUnionLabelBio]]
    matrix: BertMatrix
    tokens: List[Token]
    labels: List[BertUnionLabelBio]

    def matrix_find_prob_max(
        self, start: int, end: int
    ) -> Tuple[BertUnionLabel, float]:
        logger.debug(f"Matrix find: {start} - {end}")
        data: Dict[
            BertUnionLabel, List[int]
        ] = dict()  # { 'Actor': [1, 2], ... }
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

    def apply_fix(self: BertResult, fixes: List[EntityFix]) -> BertResult:
        new_inst = dataclasses.replace(self)
        for hyb, ali, bali, lab in fixes:
            assert len(bali) in [1, 2]
            if lab == "Both":
                lab_m, avg = self.matrix_find_prob_max(bali[0], bali[-1])
                assert is_fix_entity_label(lab_m)
                lab = lab_m

            key = bali[0]
            assert is_bert_union_label(lab)
            mapping = label_mapping_bio(lab)
            new_inst.preds[key] = mapping[0]

            if len(bali) == 2:
                for i in range(key, bali[-1] + 1):
                    new_inst.preds[i] = mapping[1]

        if fixes:
            logger.debug(f"Apply fix - Before: {self}")
            logger.debug(f"Apply fix - Fix: {fixes}")
            logger.error(f"Apply fix - After: {new_inst}")
        return new_inst
