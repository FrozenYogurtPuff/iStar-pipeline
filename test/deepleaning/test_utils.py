import logging

from src.deeplearning.entity.infer.result import BertResult
from src.deeplearning.entity.infer.utils import get_list_bio
from src.deeplearning.entity.utils.utils_metrics import (
    token_classification_report,
)

logger = logging.getLogger(__name__)


def test_token_classification_report():
    # 1.00
    trues = ["B-X", "I-X", "O", "B-Y", "I-X", "O"]
    preds = ["B-X", "I-X", "O", "B-Y", "I-X", "O"]
    logger.info(token_classification_report(trues, preds))

    # X Prec 1.0, X Recall 0.67
    # Y Prec 0.5, Y Recall 1.0
    trues = ["B-X", "I-X", "O", "B-Y", "I-X", "O"]
    preds = ["B-X", "I-X", "O", "B-Y", "I-Y", "O"]
    logger.info(token_classification_report(trues, preds))


def test_get_list_bio():
    src = [
        BertResult(["AP", "BP", "CP"], ["AT", "BT", "CT"], [[]], [], []),
        BertResult(["CP", "DP", "EP"], ["CT", "DT", "ET"], [[]], [], []),
    ]

    plist, tlist = get_list_bio(src)

    assert plist == ["AP", "BP", "CP", "CP", "DP", "EP"]
    assert tlist == ["AT", "BT", "CT", "CT", "DT", "ET"]
