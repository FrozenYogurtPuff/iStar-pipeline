from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from src.rules.dispatch import prob_bert_merge, simple_bert_merge
from pathlib import Path

from tqdm import tqdm
import logging

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
from src.rules.config import actor_plugins
from src.rules.entity.actor_plugins.include import dative_propn
from src.rules.entity.actor_plugins.include import dep as actor_dep
from src.rules.entity.actor_plugins.include import relcl_who
from src.rules.entity.actor_plugins.include import tag as actor_tag
from src.rules.entity.resource_plugins import agent_dative_adp, poss_propn
from src.rules.entity.resource_plugins import word_list as resource_word_list
from src.rules.intention.aux_slice import acl_without_to as awt_slice
from src.rules.intention.aux_slice import agent
from src.rules.intention.aux_slice import relcl as relcl_slice
from src.rules.intention.intention_plugins import acl_to
from src.utils.typing import IntentionAuxPlugins, RulePlugins
from src.rules.entity.actor_plugins.include import word_list as actor_word_list
from src.rules.entity.actor_plugins.include import ner as actor_ner
from src.rules.entity.actor_plugins.include import xcomp_ask, be_nsubj, by_sb

ex = Experiment("IP_actor_simple_merge")
ex.observers.append(MongoObserver(url='localhost:27017', db_name='sacred'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

logging.disable(logging.CRITICAL)


@ex.config
def config():
    bert_func_raw = "simple_bert_merge"
    plugins = "actor_plugins"


@ex.automain
def main(bert_func_raw, plugins, _run):
    K = 10
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

        new_pred_entities = list()
        for sent, result in zip(sents, results):
            bert_func = eval(bert_func_raw)
            funcs = eval(plugins)
            res = get_rule_fixes(
                sent, result, funcs=funcs,
                bert_func=bert_func
            )
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

    avg_data = dict()
    print('Avg')
    for key in all_data:
        avg_data[key] = dict()
        length = len(all_data[key]["p"])
        sum_p, sum_r, sum_f1 = sum(all_data[key]["p"]), sum(
            all_data[key]["r"]), sum(all_data[key]["f1"])
        avg_p, avg_r, avg_f1 = sum_p / length, sum_r / length, sum_f1 / length
        avg_data[key]["p"] = avg_p
        avg_data[key]["r"] = avg_r
        avg_data[key]["f1"] = avg_f1
        print(key, avg_p, avg_r, avg_f1, sep='\t')

    _run.all = all_data
    _run.avg = avg_data
