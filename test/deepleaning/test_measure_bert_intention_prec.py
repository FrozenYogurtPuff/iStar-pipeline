import logging
import pickle
from test.rules.utils.load_dataset import load_dataset

from src.deeplearning.infer.utils import get_list_bio, get_series_bio
from src.deeplearning.utils.utils_metrics import (
    classification_report,
    token_classification_report,
)

logger = logging.getLogger(__name__)


def test_measure_bert_intention_prec():
    data = list(load_dataset("pretrained_data/2022/task/all/split_dev.jsonl"))
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f"First items: sent {sents[0]}")
    logger.info(f"First items: label {labels[0]}")

    # wrapper = IntentionWrapper()
    # results = wrapper.process(sents, labels)
    with open("intention_split_dev.bin", "rb") as file:
        results = pickle.load(file)
    logger.info(f"First result: {results[0]}")

    pred_entities, true_entities = get_series_bio(results)

    print(classification_report(true_entities, pred_entities))

    #            precision    recall  f1-score   support
    #
    #       Aux    0.43636   0.39344   0.41379        61
    #      Cond    0.64000   0.69565   0.66667        23
    #      Core    0.66254   0.67937   0.67085       315
    #
    # micro avg    0.63027   0.63659   0.63342       399
    # macro avg    0.62666   0.63659   0.63131       399

    preds, trues = get_list_bio(results)
    print(token_classification_report(trues, preds))

    #        precision    recall  f1-score   support
    #
    #   Aux    0.61081   0.48085   0.53810       113
    #  Cond    0.73810   0.77500   0.75610        62
    #  Core    0.84214   0.89464   0.86759      1435
