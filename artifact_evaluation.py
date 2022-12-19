import argparse
import pickle
import re
import logging

import spacy

from src.deeplearning.entity.infer.utils import get_series_bio
from src.deeplearning.entity.infer.wrapper import ActorWrapper, \
    IntentionWrapper, ActorCombinedWrapper
from src.deeplearning.entity.utils.utils_metrics import classification_report
from src.deeplearning.relation.code.tasks.infer import infer_from_trained
from src.rules.config import intention_plugins
from src.rules.entity.dispatch import get_rule_fixes
from test.rules.utils.load_dataset import load_dataset


logging.disable(logging.CRITICAL)


def test_measure_bert_actor_prec():
    data = list(
        load_dataset("pretrained_data/2022_Kfold/actor/10/9/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)

    print(classification_report(true_entities, pred_entities))


def test_measure_bert_actor_rules_prec():
    data = list(
        load_dataset("pretrained_data/2022_Kfold/actor/10/0/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result)
        new_pred_entities.append(res)

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


def test_measure_bert_actor_combined_prec():
    data = list(
        load_dataset("pretrained_data/2022/actor/combined/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorCombinedWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)
    print(classification_report(true_entities, pred_entities))


def test_measure_bert_actor_combined_rules_prec():
    data = list(
        load_dataset("pretrained_data/2022/actor/combined/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorCombinedWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result)
        new_pred_entities.append(res)

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


def test_measure_bert_intention_verb_prec():
    data = list(load_dataset("pretrained_data/2022/task/verb/split_dev.jsonl"))
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = IntentionWrapper()
    results = wrapper.process(sents, labels)

    pred_entities, true_entities = get_series_bio(results)

    print(classification_report(true_entities, pred_entities))


def test_measure_bert_intention_rules_prec():
    data = list(load_dataset("pretrained_data/2022/task/verb/split_dev.jsonl"))
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = IntentionWrapper()
    results = wrapper.process(sents, labels)

    new_pred_entities = list()
    for sent, result in zip(sents, results):
        res = get_rule_fixes(sent, result, intention_plugins)
        new_pred_entities.append(res)

    pred_entities, true_entities = get_series_bio(new_pred_entities)
    print(classification_report(true_entities, pred_entities))


def test_measure_bert_relation_prec():
    args = argparse.Namespace(**dict(task='istar', train_data='./pretrained_data/2022/relation/admin.jsonl', use_pretrained_blanks=0, num_classes=4, batch_size=32, gradient_acc_steps=1, max_norm=1.0, fp16=0, num_epochs=25, lr=7e-05, model_no=0, model_size='bert-base-uncased', train=0, infer=1))
    inferer = infer_from_trained(args, detect_entities=False)
    tp, fp, tn, fn = 0, 0, 0, 0
    with open("pretrained_data/2022/relation/df_test.pkl", 'rb') as pkl_file:
        test = pickle.load(pkl_file)
        for index, row in test.iterrows():
            sents = row["sents"]
            relations = row["relations"]
            trues = row["relations_id"]     # no: 1; dependency: 0; isa: 2
            preds = inferer.infer_sentence(sents, detect_entities=False)
            if trues == 1:
                if trues == preds:
                    tn += 1
                else:
                    fp += 1
            elif preds == 1:
                # trues != 1
                fn += 1
            else:
                if trues == preds:
                    tp += 1
                else:
                    fp += 1

        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=5)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        report += row_fmt.format(
            *["Relation", r, p, f1, tp + fn], width=5, digits=5
        )
        print(report)


def find_children(token, dep=None, pos=None, tag=None, text=None):
    children_list = list()
    if not list(token.children):
        return children_list
    for t in token.children:
        flag = True
        if dep and t.dep_ != dep:
            flag = False
        if pos and t.pos != pos:
            flag = False
        if tag and t.tag != tag:
            flag = False
        if text and t.lower_ != text:
            flag = False
        if flag:
            children_list.append(t)
    return children_list


def match(stat, e11, e22):
    nsubj = list()
    obj = None

    for token in stat:
        if token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
            nsubj.append(token)
        if token.dep_ in ['xcomp', 'ccomp'] and nsubj:  # and token.head == nsubj.head:
            obj = find_children(token.head, dep='dobj')
            obj.extend(find_children(token, dep='dobj'))
            obj.extend(find_children(token, dep='pobj'))
            obj.extend(find_children(token.head, dep='pobj'))

    if nsubj and obj:
        for n in nsubj:
            for o in obj:
                if n.text in e11 and o.text in e22:
                    return True
                if n.text in e22 and o.text in e11:
                    return True


def test_measure_bert_relation_rules_prec():
    args = argparse.Namespace(**dict(task='istar', train_data='./pretrained_data/2022/relation/admin.jsonl', use_pretrained_blanks=0, num_classes=4, batch_size=32, gradient_acc_steps=1, max_norm=1.0, fp16=0, num_epochs=25, lr=7e-05, model_no=0, model_size='bert-base-uncased', train=0, infer=1))
    inferer = infer_from_trained(args, detect_entities=False)
    tp, fp, tn, fn = 0, 0, 0, 0
    nlp = spacy.load('en_core_web_lg')
    with open("pretrained_data/2022/relation/df_test.pkl", 'rb') as pkl_file:
        test = pickle.load(pkl_file)
        for index, row in test.iterrows():
            sents = row["sents"]
            relations = row["relations"]
            trues = row["relations_id"]     # no: 1; dependency: 0; isa: 2
            preds = inferer.infer_sentence(sents, detect_entities=False)

            e1 = re.search(r'\[E1](.*)\[/E1]', sents)
            if not e1:
                raise "Illegal: No e1!"
            e1 = e1.group(1)

            e2 = re.search(r'\[E2](.*)\[/E2]', sents)
            if not e2:
                raise "Illegal: No e2!"
            e2 = e2.group(1)

            raw_sent = sents.replace('[E1]', '').replace('[/E1]', '').replace('[E2]', '').replace('[/E2]', '')
            text = nlp(raw_sent)

            if match(text, e1, e2):
                preds = 0

            if trues == 1:
                if trues == preds:
                    tn += 1
                else:
                    fp += 1
            elif preds == 1:
                # trues != 1
                fn += 1
            else:
                if trues == preds:
                    tp += 1
                else:
                    fp += 1

        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=5)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        report += row_fmt.format(
            *["Relation", r, p, f1, tp + fn], width=5, digits=5
        )
        print(report)


if __name__ == '__main__':
    print("Table 2: Hybrid Method")
    print("--------------------------------------")
    print("Actor Entity")
    test_measure_bert_actor_rules_prec()
    print("--------------------------------------")
    print("Intention Entity")
    test_measure_bert_intention_rules_prec()
    print("--------------------------------------")
    print("Actor Relation")
    test_measure_bert_relation_rules_prec()
    print("--------------------------------------")
    print()
    print("Table 3: Pure BERT Method")
    print("--------------------------------------")
    print("Actor Entity")
    test_measure_bert_actor_prec()
    print("--------------------------------------")
    print("Intention Entity")
    test_measure_bert_intention_verb_prec()
    print("--------------------------------------")
    print("Actor Relation")
    test_measure_bert_relation_prec()
    print("--------------------------------------")
    print()
    print("Table 4: Combined Actor Entities")
    print("--------------------------------------")
    print("Actor Entity - BERT")
    test_measure_bert_actor_combined_prec()
    print("--------------------------------------")
    print("Actor Entity - Hybrid Method")
    test_measure_bert_actor_combined_rules_prec()
    print("--------------------------------------")
