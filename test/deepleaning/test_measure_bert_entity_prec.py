import logging
import pickle
from test.rules.utils.load_dataset import load_dataset

from src.deeplearning.infer.utils import get_series_bio
from src.deeplearning.infer.wrapper import ActorWrapper, ResourceWrapper
from src.deeplearning.utils.utils_metrics import classification_report
from src.rules.config import resource_plugins
from src.rules.entity.dispatch import get_rule_fixes

logger = logging.getLogger(__name__)


def test_token_correct():
    def get_line():
        with open("pretrained_data/2022/actor/divided/dev.txt", "r") as file:
            for li in file:
                yield li

    data = list(
        load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorWrapper()
    results = wrapper.process(sents, labels)
    yielder = get_line()

    for idx, result in enumerate(results):
        tokens = result.tokens
        trues = result.trues

        assert tokens is not None
        assert trues is not None
        assert len(tokens) == len(trues)

        for token, true in zip(tokens, trues):
            try:
                sent = next(yielder)
                if sent != f"{token} {true}\n":
                    logger.error(result)
                    logger.error(f"-{sent}")
                    logger.error(f"+{token} {true}")
            except StopIteration:
                logger.error(f"Unexpected Ending at Line #{idx}.")

        sent = next(yielder)
        if sent != "\n":
            logger.error(f"-{sent}")
            logger.error(f"+\\n")


def test_measure_bert_actor_prec():
    data = list(
        load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f"First items: sent {sents[0]}")
    logger.info(f"First items: label {labels[0]}")

    wrapper = ActorWrapper()
    results = wrapper.process(sents, labels)
    logger.info(f"First result: {results[0]}")

    pred_entities, true_entities = get_series_bio(results)

    print(classification_report(true_entities, pred_entities))

    #            precision    recall  f1-score   support
    #
    #      Role    0.80745   0.83871   0.82278       155
    #     Agent    0.93125   0.89759   0.91411       166
    #
    # micro avg    0.86916   0.86916   0.86916       321
    # macro avg    0.87147   0.86916   0.87001       321


def test_measure_bert_resource_prec():
    data = list(load_dataset("pretrained_data/2022/resource/split_dev.jsonl"))
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f"First items: sent {sents[0]}")
    logger.info(f"First items: label {labels[0]}")

    wrapper = ResourceWrapper()
    results = wrapper.process(sents, labels)
    logger.info(f"First result: {results[0]}")

    pred_entities, true_entities = get_series_bio(results)

    print(classification_report(true_entities, pred_entities))

    #            precision    recall  f1-score   support
    #
    #  Resource    0.66434   0.71788   0.69007       397
    #
    # micro avg    0.66434   0.71788   0.69007       397
    # macro avg    0.66434   0.71788   0.69007       397


def test_measure_bert_actor_rules_prec():
    data = list(
        load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f"First items: sent {sents[0]}")
    logger.info(f"First items: label {labels[0]}")

    # wrapper = ActorWrapper()
    # results = wrapper.process(sents, labels)
    with open("actor_split_dev.bin", "rb") as file:
        results = pickle.load(file)
    logger.info(f"First result: {results[0]}")

    pred_entities, true_entities = get_series_bio(results)
    print(classification_report(true_entities, pred_entities))

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result)
        new_pred_entities.append(result.apply_fix(res))

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


#            precision    recall  f1-score   support
#
#      Role    0.82237   0.80645   0.81433       155
#     Agent    0.92903   0.86747   0.89720       166
#
# micro avg    0.87622   0.83801   0.85669       321
# macro avg    0.87753   0.83801   0.85718       321


def test_measure_bert_resource_rules_prec():
    data = list(load_dataset("pretrained_data/2022/resource/split_dev.jsonl"))
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f"First items: sent {sents[0]}")
    logger.info(f"First items: label {labels[0]}")

    wrapper = ResourceWrapper()
    results = wrapper.process(sents, labels)
    logger.info(f"First result: {results[0]}")

    pred_entities, true_entities = get_series_bio(results)
    print(classification_report(true_entities, pred_entities))

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result, resource_plugins)
        new_pred_entities.append(result.apply_fix(res))

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


#            precision    recall  f1-score   support
#
#  Resource    0.64066   0.68262   0.66098       397
#
# micro avg    0.64066   0.68262   0.66098       397
# macro avg    0.64066   0.68262   0.66098       397
