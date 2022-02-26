from __future__ import annotations  # Remove that after Python 3.10

import dataclasses
import logging
from typing import List, Optional

from src.deeplearning.infer.utils import label_mapping_bio, label_mapping_de_bio
from src.utils.typing import BertEntityLabelBio, BertMatrix, Token, EntityFix, BertEntityLabel

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BertResult:
    preds: List[BertEntityLabelBio]
    trues: Optional[List[BertEntityLabelBio]]
    matrix: BertMatrix
    tokens: List[Token]
    labels: List[BertEntityLabelBio]

    def matrix_find_prob_max(self, start, end) -> BertEntityLabel:
        logger.debug(f'Matrix find: {start} - {end}')
        data = dict()  # { 'Actor': [1, 2], ... }
        for i, la in enumerate(self.labels):
            if la == 'O':
                continue
            key, _ = label_mapping_de_bio(la)
            if key in data:
                data[key].append(i)
            else:
                data[key] = [i]

        max_value, max_type = 0, None
        for key in data:
            temp_value, temp_type = 0, key
            idx = data[key]
            for tok in self.matrix[start:end + 1]:
                for i in idx:
                    temp_value += tok[i]
            if temp_value > max_value:
                max_value, max_type = temp_value, temp_type

        logger.debug(f'Matrix type: {max_type}')
        return max_type

    def apply_fix(self: BertResult, fixes: List[EntityFix]) -> BertResult:
        new_inst = dataclasses.replace(self)
        for hyb, ali, bali, lab in fixes:
            assert len(bali) in [1, 2]
            if lab == 'Both':
                lab = self.matrix_find_prob_max(bali[0], bali[-1])

            key = bali[0]
            mapping = label_mapping_bio(lab)
            new_inst.preds[key] = mapping[0]

            if len(bali) == 2:
                for i in range(key, bali[-1] + 1):
                    new_inst.preds[key] = mapping[1]

        if fixes:
            logger.debug(f'Apply fix - Before: {BertResult}')
            logger.debug(f'Apply fix - Fix: {fixes}')
            logger.debug(f'Apply fix - After: {new_inst}')
        return new_inst
