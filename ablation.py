import argparse
import pickle
import re
import logging
from pathlib import Path

import spacy
from tqdm import tqdm

from src import ROOT_DIR
from src.deeplearning.entity.infer.utils import get_series_bio
from src.deeplearning.entity.infer.wrapper import ActorWrapper, \
    IntentionWrapper, ActorCombinedWrapper
from src.deeplearning.entity.utils.utils_metrics import classification_report, compact_classification_report
from src.deeplearning.relation import kfold
from src.deeplearning.relation.code.tasks.infer import infer_from_trained
from src.rules.config import intention_plugins
from src.rules.entity.dispatch import get_rule_fixes
from test.rules.utils.load_dataset import load_dataset

logging.disable(logging.CRITICAL)

K = 10


def test_ae_bert():
    all_data = dict()
    for i in tqdm(range(K)):
        data = list(
            load_dataset(f"pretrained_data/2022_Kfold/actor/10/{i}/split_dev.jsonl")
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/actor/10/{i}/")
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/actor/10/{i}/"
        )
        label = str(
            Path(ROOT_DIR) / "pretrained_data/2022_Kfold/actor/10/labels.txt"
        )

        wrapper = ActorWrapper(data=data2, model=model, label=label)
        results = wrapper.process(sents, labels)

        pred_entities, true_entities = get_series_bio(results)
        types, ps, rs, f1s = compact_classification_report(true_entities, pred_entities)

        print(f'Fold {i}')
        for t, p, r, f1 in zip(types, ps, rs, f1s):
            print(t, p, r, f1, sep='\t')
            if t not in all_data:
                all_data[t] = dict()
                all_data[t]["p"] = list()
                all_data[t]["r"] = list()
                all_data[t]["f1"] = list()
            all_data[t]["p"].append(p)
            all_data[t]["r"].append(r)
            all_data[t]["f1"].append(f1)

    print('Avg')
    for key in all_data:
        length = len(all_data[key]["p"])
        sum_p, sum_r, sum_f1 = sum(all_data[key]["p"]), sum(
            all_data[key]["r"]), sum(all_data[key]["f1"])
        avg_p, avg_r, avg_f1 = sum_p / length, sum_r / length, sum_f1 / length
        print(key, avg_p, avg_r, avg_f1, sep='\t')


def test_ae_bert_rules():
    # CAUTION: Check `EXCLUDE=False`
    all_data = dict()
    for i in tqdm(range(K)):
        data = list(
            load_dataset(f"pretrained_data/2022_Kfold/actor/10/{i}/split_dev.jsonl")
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/actor/10/{i}/")
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/actor/10/{i}/"
        )
        label = str(
            Path(ROOT_DIR) / "pretrained_data/2022_Kfold/actor/10/labels.txt"
        )

        wrapper = ActorWrapper(data=data2, model=model, label=label)
        results = wrapper.process(sents, labels)

        new_pred_entities = list()
        for sent, result in zip(sents, results):
            res = get_rule_fixes(sent, result)
            new_pred_entities.append(res)

        pred_entities, true_entities = get_series_bio(new_pred_entities)
        types, ps, rs, f1s = compact_classification_report(true_entities, pred_entities)

        print(f'Fold {i}')
        for t, p, r, f1 in zip(types, ps, rs, f1s):
            print(t, p, r, f1, sep='\t')
            if t not in all_data:
                all_data[t] = dict()
                all_data[t]["p"] = list()
                all_data[t]["r"] = list()
                all_data[t]["f1"] = list()
            all_data[t]["p"].append(p)
            all_data[t]["r"].append(r)
            all_data[t]["f1"].append(f1)

    print('Avg')
    for key in all_data:
        length = len(all_data[key]["p"])
        sum_p, sum_r, sum_f1 = sum(all_data[key]["p"]), sum(all_data[key]["r"]), sum(all_data[key]["f1"])
        avg_p, avg_r, avg_f1 = sum_p / length, sum_r / length, sum_f1 / length
        print(key, avg_p, avg_r, avg_f1, sep='\t')


def test_ie_bert():
    all_data = dict()
    for i in tqdm(range(K)):
        data = list(
            load_dataset(f"pretrained_data/2022_Kfold/intention/10/{i}/split_dev.jsonl")
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/intention/10/{i}/")
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/intention/10/{i}/"
        )
        label = str(
            Path(ROOT_DIR) / "pretrained_data/2022_Kfold/intention/10/labels.txt"
        )

        wrapper = IntentionWrapper(data=data2, model=model, label=label)
        results = wrapper.process(sents, labels)

        pred_entities, true_entities = get_series_bio(results)
        types, ps, rs, f1s = compact_classification_report(true_entities, pred_entities)

        print(f'Fold {i}')
        for t, p, r, f1 in zip(types, ps, rs, f1s):
            print(t, p, r, f1, sep='\t')
            if t not in all_data:
                all_data[t] = dict()
                all_data[t]["p"] = list()
                all_data[t]["r"] = list()
                all_data[t]["f1"] = list()
            all_data[t]["p"].append(p)
            all_data[t]["r"].append(r)
            all_data[t]["f1"].append(f1)

    print('Avg')
    for key in all_data:
        length = len(all_data[key]["p"])
        sum_p, sum_r, sum_f1 = sum(all_data[key]["p"]), sum(
            all_data[key]["r"]), sum(all_data[key]["f1"])
        avg_p, avg_r, avg_f1 = sum_p / length, sum_r / length, sum_f1 / length
        print(key, avg_p, avg_r, avg_f1, sep='\t')


def test_ie_bert_rules():
    # CAUTION: Check `EXCLUDE=False`
    all_data = dict()
    for i in tqdm(range(K)):
        data = list(
            load_dataset(f"pretrained_data/2022_Kfold/intention/10/{i}/split_dev.jsonl")
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/intention/10/{i}/")
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/intention/10/{i}/"
        )
        label = str(
            Path(ROOT_DIR) / "pretrained_data/2022_Kfold/intention/10/labels.txt"
        )

        wrapper = IntentionWrapper(data=data2, model=model, label=label)
        results = wrapper.process(sents, labels)

        new_pred_entities = list()
        for sent, result in zip(sents, results):
            res = get_rule_fixes(sent, result, intention_plugins)
            new_pred_entities.append(res)

        pred_entities, true_entities = get_series_bio(new_pred_entities)
        types, ps, rs, f1s = compact_classification_report(true_entities, pred_entities)

        print(f'Fold {i}')
        for t, p, r, f1 in zip(types, ps, rs, f1s):
            print(t, p, r, f1, sep='\t')
            if t not in all_data:
                all_data[t] = dict()
                all_data[t]["p"] = list()
                all_data[t]["r"] = list()
                all_data[t]["f1"] = list()
            all_data[t]["p"].append(p)
            all_data[t]["r"].append(r)
            all_data[t]["f1"].append(f1)

    print('Avg')
    for key in all_data:
        length = len(all_data[key]["p"])
        sum_p, sum_r, sum_f1 = sum(all_data[key]["p"]), sum(
            all_data[key]["r"]), sum(all_data[key]["f1"])
        avg_p, avg_r, avg_f1 = sum_p / length, sum_r / length, sum_f1 / length
        print(key, avg_p, avg_r, avg_f1, sep='\t')


def test_ar_bert():
    p_all, r_all, f1_all = list(), list(), list()
    for i in range(K):
        kfold.select = i
        args = argparse.Namespace(
            **dict(task='istar', train_data='./pretrained_data/2022/relation/admin.jsonl', use_pretrained_blanks=0,
                   num_classes=4, batch_size=32, gradient_acc_steps=1, max_norm=1.0, fp16=0, num_epochs=25, lr=7e-05,
                   model_no=0, model_size='bert-base-uncased', train=0, infer=1))
        inferer = infer_from_trained(args, detect_entities=False)
        tp, fp, tn, fn = 0, 0, 0, 0
        with open(f"pretrained_data/2022_Kfold/relation/{i}/df_test.pkl", 'rb') as pkl_file:
            test = pickle.load(pkl_file)
            for index, row in test.iterrows():
                sents = row["sents"]
                relations = row["relations"]
                trues = row["relations_id"]  # no: 1; dependency: 0; isa: 2
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

            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
            print(i, p, r, f1, sep='\t')


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


def test_ar_bert_rules():
    p_all, r_all, f1_all = list(), list(), list()
    nlp = spacy.load('en_core_web_lg')
    for i in range(K):
        kfold.select = i
        args = argparse.Namespace(
            **dict(task='istar', train_data='./pretrained_data/2022/relation/admin.jsonl', use_pretrained_blanks=0,
                   num_classes=4, batch_size=32, gradient_acc_steps=1, max_norm=1.0, fp16=0, num_epochs=25, lr=7e-05,
                   model_no=0, model_size='bert-base-uncased', train=0, infer=1))
        inferer = infer_from_trained(args, detect_entities=False)
        tp, fp, tn, fn = 0, 0, 0, 0
        with open(f"pretrained_data/2022_Kfold/relation/{i}/df_test.pkl", 'rb') as pkl_file:
            test = pickle.load(pkl_file)
            for index, row in test.iterrows():
                sents = row["sents"]
                relations = row["relations"]
                trues = row["relations_id"]  # no: 1; dependency: 0; isa: 2
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

            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
            print(i, p, r, f1, sep='\t')
