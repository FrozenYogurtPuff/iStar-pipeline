import logging
from test.rules.utils.load_dataset import load_dataset

from src.deeplearning.entity.infer.utils import get_series_bio
from src.deeplearning.entity.infer.wrapper import ActorWrapper
from src.deeplearning.entity.utils.utils_metrics import classification_report
from src.rules.entity.actor_plugins.include import (
    be_nsubj,
    by_sb,
    dative_propn,
)
from src.rules.entity.actor_plugins.include import dep as actor_dep
from src.rules.entity.actor_plugins.include import ner as actor_ner
from src.rules.entity.actor_plugins.include import relcl_who
from src.rules.entity.actor_plugins.include import tag as actor_tag
from src.rules.entity.actor_plugins.include import word_list as actor_word_list
from src.rules.entity.actor_plugins.include import xcomp_ask
from src.rules.entity.dispatch import get_rule_fixes
from src.utils.typing import RulePlugin, RulePlugins


def evaluate_actor_pure_bert():
    data = list(
        load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)

    print(classification_report(true_entities, pred_entities))


def evaluate_actor_single_rule(rule: RulePlugin):
    data = list(
        load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result, (rule,))
        new_pred_entities.append(res)

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


def evaluate_actor_rules(rules: RulePlugins):
    data = list(
        load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorWrapper()
    results = wrapper.process(sents, labels)

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result, rules)
        new_pred_entities.append(res)

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)

    actor_plugins = (
        dative_propn,
        relcl_who,
        actor_tag,
        actor_dep,
        actor_word_list,
        actor_ner,
        xcomp_ask,
        be_nsubj,
        by_sb,
    )

    # for ap in actor_plugins:
    #     print(ap.__name__)
    #     evaluate_actor_single_rule(ap)

    # PURE
    # evaluate_actor_pure_bert()
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # Only Include
    # dative_propn -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # relcl_who -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # actor_tag -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # actor_dep -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # actor_word_list -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # ner -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # xcomp_ask -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # be_nsubj -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # by_sb -
    #             precision    recall  f1-score   support
    #     Agent    0.93125   0.89759   0.91411       166
    #      Role    0.80745   0.83871   0.82278       155

    # Include + Exclude [-1]noun
    #           precision    recall  f1-score   support
    #   Agent    0.95513   0.89759   0.92547       166
    #    Role    0.83117   0.82581   0.82848       155
