import argparse
import pickle
import re
import logging
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm
from sklearn.metrics import classification_report as cr

from src import ROOT_DIR
from src.deeplearning.entity.infer.result import BertResult
from src.deeplearning.entity.infer.utils import get_series_bio
from src.deeplearning.entity.infer.wrapper import ActorWrapper, \
    IntentionWrapper, ActorCombinedWrapper
from src.deeplearning.entity.utils.utils_metrics import classification_report, \
    compact_classification_report
from src.deeplearning.relation import kfold
from src.deeplearning.relation.code.tasks.infer import infer_from_trained
# from src.rules.config import intention_plugins
# from src.rules.entity.actor_plugins.include import xcomp_ask, be_nsubj, by_sb
from src.rules.entity.dispatch import get_rule_fixes
# from test.rules.inspect.entity_rules import dative_propn, relcl_who, tag_base, \
#     ner, prep_sb, acomp_template, acl_to, able_to, nsubjpass_head
from test.rules.inspect.relation_rules import default, agent_pobj, \
    conj_exclude, nsubj_attr, consists, nsubj_pobj
from test.rules.utils.load_dataset import load_dataset


logging.disable(logging.CRITICAL)

K = 10


# 用来支持 AE 测试里 Agent、Role 转 Actor 的
def transtype(results: list[BertResult]) -> list[BertResult]:
    ret = list(results)
    for result in ret:
        result.labels = ['O', 'B-Actor', 'I-Actor']
        for j in [result.preds, result.trues]:
            for i in range(len(result.preds)):
                if j[i] in ['B-Agent', 'B-Role']:
                    j[i] = 'B-Actor'
                elif j[i] in ['I-Agent', 'I-Role']:
                    j[i] = 'I-Actor'
    return ret


# AE 深度测试
def test_ae_bert():
    all_data = dict()
    for i in tqdm(range(K)):
        data = list(
            load_dataset(
                f"pretrained_data/2022_Kfold/actor/10/{i}/split_dev.jsonl")
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(
            Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/actor/10/{i}/")
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/actor/10/{i}/"
        )
        label = str(
            Path(ROOT_DIR) / "pretrained_data/2022_Kfold/actor/10/labels.txt"
        )

        wrapper = ActorWrapper(data=data2, model=model, label=label)
        results = wrapper.process(sents, labels)
        # with open(f"cache/ae_bert_{i}.bin", "rb") as file:
        #     results = pickle.load(file)

        # results = transtype(results)

        pred_entities, true_entities = get_series_bio(results)
        types, ps, rs, f1s = compact_classification_report(true_entities,
                                                           pred_entities)

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


# AE 深度+规则测试
def test_ae_bert_rules():
    # action_plugins_new = (
    #     dative_propn,
    #     relcl_who,
    #     tag_base,
    #     ner,
    #     prep_sb,
    # )

    # CAUTION: Check `EXCLUDE=False`
    all_data = dict()
    for i in tqdm(range(K)):
        data = list(
            load_dataset(
                f"pretrained_data/2022_Kfold/actor/10/{i}/split_dev.jsonl")
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(
            Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/actor/10/{i}/")
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/actor/10/{i}/"
        )
        label = str(
            Path(ROOT_DIR) / "pretrained_data/2022_Kfold/actor/10/labels.txt"
        )

        wrapper = ActorWrapper(data=data2, model=model, label=label)
        results = wrapper.process(sents, labels)
        # with open(f"cache/ae_bert_{i}.bin", "rb") as file:
        #     results = pickle.load(file)

        new_pred_entities = list()
        for sent, result in zip(sents, results):
            res = get_rule_fixes(sent, result, desc="AE")
            new_pred_entities.append(res)

        # new_pred_entities = transtype(new_pred_entities)

        pred_entities, true_entities = get_series_bio(new_pred_entities)
        types, ps, rs, f1s = compact_classification_report(true_entities,
                                                           pred_entities)

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


def test_ie_bert():
    all_data = dict()
    for i in tqdm(range(K)):
        data = list(
            load_dataset(
                f"pretrained_data/2022_Kfold/intention/10/{i}/split_dev.jsonl")
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(
            Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/intention/10/{i}/")
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/intention/10/{i}/"
        )
        label = str(
            Path(
                ROOT_DIR) / "pretrained_data/2022_Kfold/intention/10/labels.txt"
        )

        wrapper = IntentionWrapper(data=data2, model=model, label=label)
        results = wrapper.process(sents, labels)
        # with open(f"cache/ie_bert_{i}.bin", "rb") as file:
        #     results = pickle.load(file)

        pred_entities, true_entities = get_series_bio(results)
        types, ps, rs, f1s = compact_classification_report(true_entities,
                                                           pred_entities)

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
    # intention_plugins_new = (
    #     acl_to,
    # )

    # CAUTION: Check `EXCLUDE=False`
    all_data = dict()
    for i in tqdm(range(K)):
        data = list(
            load_dataset(
                f"pretrained_data/2022_Kfold/intention/10/{i}/split_dev.jsonl")
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(
            Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/intention/10/{i}/")
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/intention/10/{i}/"
        )
        label = str(
            Path(
                ROOT_DIR) / "pretrained_data/2022_Kfold/intention/10/labels.txt"
        )

        wrapper = IntentionWrapper(data=data2, model=model, label=label)
        results = wrapper.process(sents, labels)
        # with open(f"cache/ie_bert_{i}.bin", "rb") as file:
        #     results = pickle.load(file)

        new_pred_entities = list()
        for sent, result in zip(sents, results):
            res = get_rule_fixes(sent, result, desc="IE")
            new_pred_entities.append(res)

        pred_entities, true_entities = get_series_bio(new_pred_entities)
        types, ps, rs, f1s = compact_classification_report(true_entities,
                                                           pred_entities)

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
    preds_list, trues_list = list(), list()
    with open("cache/ar_dict.bin", "rb") as file:
        bert_dict = pickle.load(file)

    for i in range(K):
        kfold.select = i
        args = argparse.Namespace(
            **dict(task='istar',
                   train_data='./pretrained_data/2022/relation/admin.jsonl',
                   use_pretrained_blanks=0,
                   num_classes=4, batch_size=32, gradient_acc_steps=1,
                   max_norm=1.0, fp16=0, num_epochs=25, lr=7e-05,
                   model_no=0, model_size='bert-base-uncased', train=0,
                   infer=1))
        inferer = infer_from_trained(args, detect_entities=False)
        tp, fp, tn, fn = 0, 0, 0, 0
        with open(f"pretrained_data/2022_Kfold/relation/{i}/df_test.pkl",
                  'rb') as pkl_file:
            test = pickle.load(pkl_file)
            for index, row in test.iterrows():
                sents = row["sents"]
                relations = row["relations"]
                trues = row["relations_id"]  # no: 1; dependency: 0; isa: 2; part-of: 3
                # preds = inferer.infer_sentence(sents, detect_entities=False)
                preds = bert_dict[sents]

                preds_list.append(preds)
                trues_list.append(trues)

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
            p_all.append(p)
            r_all.append(r)
            f1_all.append(f1)
            print(i, p, r, f1, sep='\t')

    print(sum(p_all) / K, sum(r_all) / K, sum(f1_all) / K)
    print(cr(trues_list, preds_list, digits=8))
    # with open("ar_dict.bin", "wb") as file:
    #     pickle.dump(bert_dict, file)


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
        if token.dep_ in ['xcomp',
                          'ccomp'] and nsubj:  # and token.head == nsubj.head:
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
    with open("cache/ar_dict.bin", "rb") as file:
        ar_dict = pickle.load(file)
    ar_rules = (
        conj_exclude,
        nsubj_attr,
        consists,
        agent_pobj,
        nsubj_pobj,
    )
    p_all, r_all, f1_all = list(), list(), list()
    preds_list, trues_list = list(), list()
    nlp = spacy.load('en_core_web_trf')
    for i in range(K):
        kfold.select = i
        args = argparse.Namespace(
            **dict(task='istar',
                   train_data='./pretrained_data/2022/relation/admin.jsonl',
                   use_pretrained_blanks=0,
                   num_classes=4, batch_size=32, gradient_acc_steps=1,
                   max_norm=1.0, fp16=0, num_epochs=25, lr=7e-05,
                   model_no=0, model_size='bert-base-uncased', train=0,
                   infer=1))

        # inferer = infer_from_trained(args, detect_entities=False)
        tp, fp, tn, fn = 0, 0, 0, 0
        with open(f"pretrained_data/2022_Kfold/relation/{i}/df_test.pkl",
                  'rb') as pkl_file:
            test = pickle.load(pkl_file)
            for index, row in test.iterrows():
                sents = row["sents"]
                relations = row["relations"]
                trues = row["relations_id"]  # no: 1; dependency: 0; isa: 2
                # try:
                preds = ar_dict[sents]
                # except KeyError:
                #     preds = inferer.infer_sentence(sents, detect_entities=False)

                e1_raw = re.search(r'\[E1](.*)\[/E1]', sents).group(1)
                e2_raw = re.search(r'\[E2](.*)\[/E2]', sents).group(1)
                e1_idx, e2_idx = None, None
                for idx, tok in enumerate(sents.split(' ')):
                    if '[E1]' in tok:
                        e1_idx = idx
                    if '[E2]' in tok:
                        e2_idx = idx
                assert e1_idx is not None
                assert e2_idx is not None
                sent = nlp(
                    sents.replace(
                        '[E1]', ''
                    ).replace(
                        '[/E1]', ''
                    ).replace(
                        '[E2]', ''
                    ).replace(
                        '[/E2]', ''
                    )
                )[:]
                matcher = PhraseMatcher(nlp.vocab, attr="ORTH")
                matcher.add("e1", [nlp(e1_raw)])
                matches = matcher(sent, as_spans=True)
                check = True
                if not matches:
                    check = False
                else:
                    e1 = min(matches, key=lambda t: abs(t.start - e1_idx))
                matcher.remove("e1")
                matcher.add("e2", [nlp(e2_raw)])
                matches = matcher(sent, as_spans=True)
                if not matches:
                    check = False
                else:
                    e2 = min(matches, key=lambda t: abs(t.start - e2_idx))

                if check:
                    for func in ar_rules:
                        result = func(sent, e1, e2)
                        if result:
                            preds = result

                preds_list.append(preds)
                trues_list.append(trues)

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
            p_all.append(p)
            r_all.append(r)
            f1_all.append(f1)
            print(i, p, r, f1, sep='\t')

    print(sum(p_all) / K, sum(r_all) / K, sum(f1_all) / K)
    print(cr(trues_list, preds_list, digits=8))


def test_ar_rules_precision():
    nlp = spacy.load('en_core_web_trf')
    for i in range(K):
        tp, fp, tn, fn = 0, 0, 0, 0
        with open(f"pretrained_data/2022_Kfold/relation/{i}/df_test.pkl",
                  'rb') as pkl_file:
            test = pickle.load(pkl_file)
            for index, row in test.iterrows():
                sents = row["sents"]
                relations = row["relations"]
                trues = row["relations_id"]  # no: 1; dependency: 0; isa: 2
                preds = 1
                e1 = re.search(r'\[E1](.*)\[/E1]', sents)
                if not e1:
                    raise "Illegal: No e1!"
                e1 = e1.group(1)

                e2 = re.search(r'\[E2](.*)\[/E2]', sents)
                if not e2:
                    raise "Illegal: No e2!"
                e2 = e2.group(1)

                raw_sent = sents.replace('[E1]', '').replace('[/E1]',
                                                             '').replace(
                    '[E2]', '').replace('[/E2]', '')
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

            p = tp / (tp + fp) if tp + fp != 0 else 0
            r = tp / (tp + fn) if tp + fn != 0 else 0
            f1 = 2 * p * r / (p + r) if p + r != 0 else 0
            print(i, p, r, f1, sep='\t')


if __name__ == '__main__':
    # AE model
    test_ae_bert()
    # AE model + rules
    test_ae_bert_rules()
    # IE model
    test_ie_bert()
    # IE model + rules
    test_ie_bert_rules()
    # AR model
    test_ar_bert()
    # AR model + rules
    test_ar_bert_rules()
