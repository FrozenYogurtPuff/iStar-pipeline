import json
import logging
from pathlib import Path

from src import ROOT_DIR
from src.deeplearning.infer.utils import get_series_bio
from src.deeplearning.infer.wrapper import infer_wrapper
from src.deeplearning.utils.utils_metrics import classification_report
from src.rules.entity.dispatch import get_rule_fixes

logger = logging.getLogger(__name__)


def load_dataset():
    with open(Path(ROOT_DIR).joinpath('pretrained_data/entity_ar_r_combined/split_dev.jsonl'), 'r') as j:
        for idx, line in enumerate(j):
            a = json.loads(line)
            sent = a['text']
            anno = list()
            for label in a['labels']:
                s, e, lab = label
                anno.append((s, e, lab))
            yield idx, sent, anno


def test_measure_bert_prec():
    data = list(load_dataset())
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f'First items: sent {sents[0]}')
    logger.info(f'First items: label {labels[0]}')

    results = infer_wrapper('Entity', sents, labels)
    logger.info(f'First result: {results[0]}')

    pred_entities, true_entities = get_series_bio(results)

    print(classification_report(true_entities, pred_entities))

    """
                   precision    recall  f1-score   support
        
         Resource    0.52677   0.54545   0.53595       451
            Actor    0.66933   0.80449   0.73071       312
        
        micro avg    0.59026   0.65138   0.61931       763
        macro avg    0.58506   0.65138   0.61559       763
    """


def test_measure_bert_simple_prec():
    data = list(load_dataset())
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]
    logger.info(f'First items: sent {sents[0]}')
    logger.info(f'First items: label {labels[0]}')

    results = infer_wrapper('Entity', sents, labels)
    logger.info(f'First result: {results[0]}')

    pred_entities, true_entities = get_series_bio(results)
    print(classification_report(true_entities, pred_entities))

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result)
        new_pred_entities.append(result.apply_fix(res))

    pred_entities, true_entities = get_series_bio(results)
    print(classification_report(true_entities, pred_entities))

