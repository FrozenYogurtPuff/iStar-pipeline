import logging
from test.rules.utils.load_dataset import load_dataset

logger = logging.getLogger(__name__)


def test_measure_bert_intention_prec():
    data = list(
        load_dataset(
            "pretrained_data/task_core_aux_cond/k-fold/combine/2/dev.jsonl"
        )
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f"First items: sent {sents[0]}")
    logger.info(f"First items: label {labels[0]}")

    # results = infer_wrapper("Intention", sents, labels)  # TODO
    # logger.info(f"First result: {results[0]}")
    #
    # pred_entities, true_entities = get_series_bio(results)

    # print(classification_report(true_entities, pred_entities))

    #            precision    recall  f1-score   support
    #
    #       Aux    0.37705   0.51111   0.43396        45
    #      Core    0.62633   0.70120   0.66165       251
    #      Cond    0.73684   0.73684   0.73684        19
    #
    # micro avg    0.59003   0.67619   0.63018       315
    # macro avg    0.59739   0.67619   0.63366       315


# TODO: 用规则救救Aux，想想办法，都有什么规则
