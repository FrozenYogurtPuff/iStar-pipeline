import logging

from src.deeplearning.entity.infer.utils import get_series_bio
from src.deeplearning.entity.infer.wrapper import IntentionWrapper
from src.deeplearning.entity.utils.utils_metrics import classification_report
from src.rules.entity.dispatch import get_rule_fixes
from src.rules.intention.intention_plugins import xcomp_to, acl_to, \
    acl_without_to, relcl, pcomp_ing, advcl
from src.utils.typing import RulePlugin, RulePlugins
from test.rules.utils.load_dataset import load_dataset


def evaluate_intention_pure_bert():
    data = list(
        load_dataset("pretrained_data/2022/task/verb/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = IntentionWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)

    print(classification_report(true_entities, pred_entities))


def evaluate_intention_single_rule(rule: RulePlugin):
    data = list(
        load_dataset("pretrained_data/2022/task/verb/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = IntentionWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result, (rule,))
        new_pred_entities.append(res)

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


def evaluate_intention_rules(rules: RulePlugins):
    data = list(
        load_dataset("pretrained_data/2022/task/verb/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = IntentionWrapper()
    results = wrapper.process(sents, labels)

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result, rules)
        new_pred_entities.append(res)

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)

    # evaluate_intention_pure_bert()
    #               precision    recall  f1-score   support
    #       Aux    0.73494   0.85915   0.79221        71
    #      Cond    0.90909   0.80000   0.85106        25
    #      Core    0.92464   0.94100   0.93275       339

    intention_plugins: RulePlugins = (
        # xcomp_to,
        acl_to,
        # acl_without_to,
        # relcl,
        pcomp_ing,
        advcl,
    )

    # for ip in intention_plugins:
    #     print(ip.__name__)
    #     evaluate_intention_single_rule(ip)

    # xcomp_to
    #               precision    recall  f1-score   support
    #       Aux    0.73171   0.84507   0.78431        71
    #      Cond    0.90909   0.80000   0.85106        25
    #      Core    0.92197   0.94100   0.93139       339

    # acl_to
    #               precision    recall  f1-score   support
    #       Aux    0.73494   0.85915   0.79221        71
    #      Cond    0.90909   0.80000   0.85106        25
    #      Core    0.92464   0.94100   0.93275       339

    # acl_without_to
    #               precision    recall  f1-score   support
    #       Aux    0.72619   0.85915   0.78710        71
    #      Cond    0.90909   0.80000   0.85106        25
    #      Core    0.92464   0.94100   0.93275       339

    # relcl
    #               precision    recall  f1-score   support
    #       Aux    0.70115   0.85915   0.77215        71
    #      Cond    0.90909   0.80000   0.85106        25
    #      Core    0.92464   0.94100   0.93275       339

    # pcomp_ing -
    # advcl -

    # evaluate_intention_rules(intention_plugins) + [-1]VB/NN
    #              precision    recall  f1-score   support
    #       Aux    0.73494   0.85915   0.79221        71
    #      Cond    0.95238   0.80000   0.86957        25
    #      Core    0.93275   0.94100   0.93686       339
