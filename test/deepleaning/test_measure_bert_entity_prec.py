import logging
from test.rules.utils.load_dataset import load_dataset

from src.deeplearning.infer.utils import get_series_bio
from src.deeplearning.infer.wrapper import ActorWrapper
from src.deeplearning.utils.utils_metrics import classification_report

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

    #                precision    recall  f1-score   support
    #
    #         Actor    0.66933   0.80449   0.73071       312
    #      Resource    0.52677   0.54545   0.53595       451
    #
    #     micro avg    0.59026   0.65138   0.61931       763
    #     macro avg    0.58506   0.65138   0.61559       763


def test_measure_bert_simple_prec():
    data = list(
        load_dataset("pretrained_data/entity_ar_r_combined/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f"First items: sent {sents[0]}")
    logger.info(f"First items: label {labels[0]}")

    # results = infer_wrapper("Entity", sents, labels)  # TODO
    # logger.info(f"First result: {results[0]}")
    #
    # pred_entities, true_entities = get_series_bio(results)
    # print(classification_report(true_entities, pred_entities))
    #
    # new_pred_entities = list()
    # for sent, result in zip(sents, results):
    #     res = get_rule_fixes(sent, result)
    #     new_pred_entities.append(result.apply_fix(res))
    #
    # pred_entities, true_entities = get_series_bio(new_pred_entities)
    # print(classification_report(true_entities, pred_entities))


# simple[?][0] precision    recall  f1-score   support
#     Actor    0.59951   0.78205   0.67872       312
#  Resource    0.47645   0.58315   0.52443       451

# micro avg    0.52868   0.66448   0.58885       763
# macro avg    0.52677   0.66448   0.58752       763


# simple[0][0] precision    recall  f1-score   support
#     Actor    0.60345   0.78526   0.68245       312
#  Resource    0.47227   0.58537   0.52277       451
#
# micro avg    0.52746   0.66710   0.58912       763
# macro avg    0.52591   0.66710   0.58807       763


# prob soft, filter 'Both' and prob < 0.5
# prob hard, filter prob < 0.5
# soft[0][0]  precision    recall  f1-score   support
#
#     Actor    0.61461   0.78205   0.68829       312
#  Resource    0.50000   0.54767   0.52275       451
#
# micro avg    0.55107   0.64351   0.59371       763
# macro avg    0.54687   0.64351   0.59044       763


# word_list `user` -> both
# hard[0][0] precision    recall  f1-score   support
#
#     Actor    0.66578   0.79808   0.72595       312
#  Resource    0.54167   0.54767   0.54465       451
#
# micro avg    0.59759   0.65007   0.62272       763
# macro avg    0.59242   0.65007   0.61879       763


# new inspect rules > .85
#            precision    recall  f1-score   support
#
#     Actor    0.71751   0.81410   0.76276       312
#  Resource    0.53209   0.56984   0.55032       451
#
# micro avg    0.61051   0.66972   0.63875       763
# macro avg    0.60791   0.66972   0.63719       763

# new inspect rules > .9 (~↑)
#            precision    recall  f1-score   support
#
#     Actor    0.71348   0.81410   0.76048       312
#  Resource    0.56332   0.57206   0.56766       451
#
# micro avg    0.62899   0.67104   0.64933       763
# macro avg    0.62472   0.67104   0.64650       763

# new inspect rules > .9 disable tag (~)
#            precision    recall  f1-score   support
#
#     Actor    0.71508   0.82051   0.76418       312
#  Resource    0.56416   0.56541   0.56478       451
#
# micro avg    0.63086   0.66972   0.64971       763
# macro avg    0.62587   0.66972   0.64632       763

# new inspect rules > .9 disable dep (↓)
#            precision    recall  f1-score   support
#
#     Actor    0.70670   0.81090   0.75522       312
#  Resource    0.56018   0.56763   0.56388       451
#
# micro avg    0.62454   0.66710   0.64512       763
# macro avg    0.62009   0.66710   0.64212       763

# new inspect rules > .9 disable ner (↓)
#            precision    recall  f1-score   support
#
#     Actor    0.71348   0.81410   0.76048       312
#  Resource    0.56087   0.57206   0.56641       451
#
# micro avg    0.62745   0.67104   0.64851       763
# macro avg    0.62328   0.67104   0.64577       763

# new inspect rules > .9 with conjuncts add base fix (↑)
#            precision    recall  f1-score   support
#
#     Actor    0.71549   0.81410   0.76162       312
#  Resource    0.56304   0.57428   0.56861       451
#
# micro avg    0.62945   0.67235   0.65019       763
# macro avg    0.62538   0.67235   0.64753       763
