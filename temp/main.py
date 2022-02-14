import re
import json

import en_core_web_lg

from src.deeplearning.infer import predict_task as bert_task
from src.deeplearning.infer import predict_entity as bert_entity
from dependency import pred_entity as dep_entity
from dependency import pred_task as dep_task
from dependency import simple_noun_chunks as simp_entity
from bert_util import get_entities_bio_with_prob, calc_specific_prob
import spacy_alignments as tokenizations


entity_labels = ['O', 'B-Actor', 'I-Actor', 'B-Resource', 'I-Resource']
task_labels = ['O', 'B-Intention', 'I-Intention']

# sent = 'Teacher Y asks student A to fill out a loan form and write down the following: information ' \
#        'about the teacher in the classroom, the reason for borrowing the classroom, and the time ' \
#        'for borrowing the classroom. '

nlp = en_core_web_lg.load()


def sent_tokenize(sent):
    return list(filter(str.split, re.split('([,|.|?|!|"|:|(|)|/| ])', sent)))


def get_bert_task(sent):
    data = [{'sent': sent}]
    return bert_task(data)


def get_bert_entity(sent):
    data = [{'sent': sent}]
    return bert_entity(data)

sents = list()
with open('../pretrained_data/entity_ar_r_combined/split_dev.jsonl', 'r') as dev:
    ground = set()
    chunk_data = set()
    ablation_bert = set()
    ablation_dep = set()
    for idxx, line in enumerate(dev):
        a = json.loads(line)
        sent = a['text']
        anno = a['labels']
        res = nlp(sent)
        for an in anno:
            strs = res.char_span(an[0], an[1])
            if strs:
                ground.add((an[2], idxx * 100 + strs.start, idxx * 100 + strs.end))

        # BERT Entity
        bert_entity_result = list()
        bert_entity_prob = list()
        bert_entity_token = list()

        bert_entity_pred_list, bert_entity_matrix, bert_entity_tokens = get_bert_entity(sent)
        for idx in range(len(bert_entity_pred_list)):
            pred_list = bert_entity_pred_list[idx]
            matrix = bert_entity_matrix[idx]
            tokens = bert_entity_tokens[idx]
            res, prob = get_entities_bio_with_prob(pred_list, matrix, entity_labels)
            bert_entity_result.append(res)
            bert_entity_prob.append(prob)
            bert_entity_token.append(bert_entity_tokens)

        # Dependency Entity
        dep_entity_result, dep_token, dep_entity_pred_list = dep_entity(sent)

        # Noun chunks
        simp_pred_list = simp_entity(sent)

        # Rule combined
        rule_pred_list = list()
        for idx in range(len(dep_entity_pred_list)):
            cur_list = sorted(dep_entity_pred_list[idx] + simp_pred_list[idx], key=lambda x: x[0])
            cur_idx = 0
            while cur_idx < len(cur_list) - 1:
                cur_entity = cur_list[cur_idx]
                next_entity = cur_list[cur_idx + 1]
                if next_entity[1] <= cur_entity[1]:
                    cur_list.pop(cur_idx + 1)
                cur_idx += 1
            rule_pred_list.append(cur_list)


        # mixing entity
        def _create_mixing():
            ret = dict()
            ret['selection'] = list()
            ret['optional'] = list()
            return ret


        def _map_a2b(a2b, a_start, a_end=None):
            start = a2b[a_start]
            if isinstance(start, list):
                start = start[0]
            if a_end is not None:
                end = a2b[a_end]
                if isinstance(end, list):
                    end = end[-1]
                return start, end
            return start


        if len(bert_entity_result) != len(rule_pred_list):
            # TODO: unexpected multi sentence
            continue
        mixing_entity = list()
        for idx in range(len(bert_entity_result)):
            cur_mix = _create_mixing()
            bert = bert_entity_result[idx]
            rule = rule_pred_list[idx]
            b2r, r2b = tokenizations.get_alignments(bert_entity_tokens[idx], dep_token[idx])
            # assert len(bert_entity_matrix[idx]) == len(dep_token[idx])
            for sub_idx, br in enumerate(bert):
                b_type = br[0]
                b_start, b_end = _map_a2b(b2r, br[1], br[2])
                b_prob = bert_entity_prob[idx][sub_idx]
                status = 1
                # Ablation
                ablation_bert.add((b_type, idxx * 100 + b_start, idxx * 100 + b_end + 1))
                # TODO: 区分核心识别与非核心识别区域
                for r in rule:
                    r_start = r[0]
                    r_end = r[1]
                    # Ablation
                    ablation_dep.add((idxx * 100 + r_start, idxx * 100 + r_end + 1))
                    if b_start <= r_start and b_end >= r_end:
                        rule.remove(r)
                        status = 2
                    elif b_start >= r_start and b_end <= r_end:
                        b_start, b_end = r_start, r_end
                        rule.remove(r)
                        status = 2
                if status == 2:
                    cur_mix['selection'].append((b_type, b_start, b_end))
                elif status == 1:
                    cur_mix['optional'].append((b_type, b_start, b_end, b_prob))
            for r in rule:
                r_start = r[0]
                r_end = r[1]
                r2b_start, r2b_end = _map_a2b(r2b, r_start, r_end)
                actor_prob = calc_specific_prob(bert_entity_matrix[idx], entity_labels, 'Actor', r2b_start, r2b_end)
                resource_prob = calc_specific_prob(bert_entity_matrix[idx], entity_labels, 'Resource', r2b_start, r2b_end)
                r_type = 'Actor' if (actor_prob > resource_prob) else 'Resource'
                r_prob = max(actor_prob, resource_prob)
                cur_mix['optional'].append((r_type, r_start, r_end, r_prob))
            mixing_entity.append(cur_mix)
        # print(mixing_entity)
        for result in mixing_entity:
            rest = result['selection']
            opt = result['optional']
            for r in rest:
                chunk_data.add((r[0], idxx * 100 + r[1], idxx * 100 + r[2] + 1))
            for o in opt:
                if o[3] > 0.9:
                    chunk_data.add((o[0], idxx * 100 + o[1], idxx * 100 + o[2] + 1))
    # ablation 为了与无类型的纯依存比较
    ground_nolabel = set([('O', g[1], g[2]) for g in ground])
    print(chunk_data)

# BERT Intention
bert_task_result = list()
bert_task_prob = list()
bert_task_token = list()

bert_task_pred_list, bert_task_matrix, bert_task_tokens = get_bert_task(sent)
for idx in range(len(bert_task_pred_list)):
    pred_list = bert_task_pred_list[idx]
    matrix = bert_task_matrix[idx]
    tokens = bert_task_tokens[idx]
    res, prob = get_entities_bio_with_prob(pred_list, matrix, task_labels)
    bert_task_result.append(res)
    bert_task_prob.append(prob)
    bert_task_token.append(bert_task_tokens)

# Dependency Intention
dep_task_result = dep_task(sent)

print(bert_task_result)
print(dep_task_result)
