# import logging
# import pickle
# from pathlib import Path
# from test.rules.utils.load_dataset import load_dataset
#
# from src import ROOT_DIR
# from src.deeplearning.entity.infer.utils import get_list_bio, get_series_bio
# from src.deeplearning.entity.infer.wrapper import (
#     ActorWrapper,
#     IntentionWrapper,
#     ResourceWrapper,
# )
# from src.deeplearning.entity.utils.utils_metrics import (
#     classification_report,
#     token_classification_report,
# )
# # from src.rules.config import intention_plugins, resource_plugins
# from src.rules.entity.dispatch import get_rule_fixes
#
# logger = logging.getLogger(__name__)
#
#
# def test_token_correct():
#     def get_line():
#         with open("pretrained_data/2022/actor/divided/dev.txt", "r") as file:
#             for li in file:
#                 yield li
#
#     data = list(
#         load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
#     )
#     sents = [d[1] for d in data]
#     labels = [d[2] for d in data]
#
#     wrapper = ActorWrapper()
#     results = wrapper.process(sents, labels)
#     yielder = get_line()
#
#     for idx, result in enumerate(results):
#         tokens = result.tokens
#         trues = result.trues
#
#         assert tokens is not None
#         assert trues is not None
#         assert len(tokens) == len(trues)
#
#         for token, true in zip(tokens, trues):
#             try:
#                 sent = next(yielder)
#                 if sent != f"{token} {true}\n":
#                     logger.error(result)
#                     logger.error(f"-{sent}")
#                     logger.error(f"+{token} {true}")
#             except StopIteration:
#                 logger.error(f"Unexpected Ending at Line #{idx}.")
#
#         sent = next(yielder)
#         if sent != "\n":
#             logger.error(f"-{sent}")
#             logger.error(f"+\\n")
#
#
# def test_measure_bert_actor_prec():
#     total_result = list()
#     for i in range(1):
#         data = list(
#             load_dataset(
#                 f"pretrained_data/2022_Kfold/actor/{i}/split_dev.jsonl"
#             )
#         )
#         sents = [d[1] for d in data]
#         labels = [d[2] for d in data]
#         logger.info(f"First items: sent {sents[0]}")
#         logger.info(f"First items: label {labels[0]}")
#
#         data2 = str(Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/actor/{i}/")
#         model = str(
#             Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/actor/{i}/output/"
#         )
#         label = str(
#             Path(ROOT_DIR) / "pretrained_data/2022_Kfold/actor/labels.txt"
#         )
#
#         wrapper = ActorWrapper(data=data2, model=model, label=label)
#         results = wrapper.process(sents, labels)
#         logger.info(f"First result: {results[0]}")
#
#         total_result.extend(results)
#
#     pred_entities, true_entities = get_series_bio(total_result)
#     print(classification_report(true_entities, pred_entities))
#
#     # Previous Actor
#     #            precision    recall  f1-score   support
#     #
#     #      Role    0.80745   0.83871   0.82278       155
#     #     Agent    0.93125   0.89759   0.91411       166
#     #
#     # micro avg    0.86916   0.86916   0.86916       321
#     # macro avg    0.87147   0.86916   0.87001       321
#
#
# def test_measure_bert_resource_prec():
#     data = list(load_dataset("pretrained_data/2022/resource/split_dev.jsonl"))
#     sents = [d[1] for d in data]
#     labels = [d[2] for d in data]
#     logger.info(f"First items: sent {sents[0]}")
#     logger.info(f"First items: label {labels[0]}")
#
#     wrapper = ResourceWrapper()
#     results = wrapper.process(sents, labels)
#     logger.info(f"First result: {results[0]}")
#
#     pred_entities, true_entities = get_series_bio(results)
#
#     print(classification_report(true_entities, pred_entities))
#
#     #            precision    recall  f1-score   support
#     #
#     #  Resource    0.66434   0.71788   0.69007       397
#     #
#     # micro avg    0.66434   0.71788   0.69007       397
#     # macro avg    0.66434   0.71788   0.69007       397
#
#
# def test_measure_bert_actor_rules_prec():
#     data = list(
#         load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
#     )
#     sents = [d[1] for d in data]
#     labels = [d[2] for d in data]
#     logger.info(f"First items: sent {sents[0]}")
#     logger.info(f"First items: label {labels[0]}")
#
#     # wrapper = ActorWrapper()
#     # results = wrapper.process(sents, labels)
#     with open("actor_split_dev.bin", "rb") as file:
#         results = pickle.load(file)
#     logger.info(f"First result: {results[0]}")
#
#     # pred_entities, true_entities = get_series_bio(results)
#     # print(classification_report(true_entities, pred_entities))
#     #
#     # preds, trues = get_list_bio(results)
#     # print(token_classification_report(trues, preds))
#
#     new_pred_entities = list()
#     for sent, result in zip(sents, results):
#         res = get_rule_fixes(sent, result)
#         new_pred_entities.append(res)
#
#     pred_entities, true_entities = get_series_bio(new_pred_entities)
#     print(classification_report(true_entities, pred_entities))
#
#     preds, trues = get_list_bio(new_pred_entities)
#     print(token_classification_report(trues, preds))
#
#
# #            precision    recall  f1-score   support
# #
# #     Agent    0.93168   0.90361   0.91743       166
# #      Role    0.80745   0.83871   0.82278       155
# #
# #
# #        precision    recall  f1-score   support
# #
# # Agent    0.91706   0.91059   0.91381       387
# #  Role    0.81864   0.85752   0.83763       325
#
#
# def test_measure_bert_actor_combined_rules_prec():
#     data = list(
#         load_dataset("pretrained_data/2022/actor/combined/split_dev.jsonl")
#     )
#     sents = [d[1] for d in data]
#     labels = [d[2] for d in data]
#     logger.info(f"First items: sent {sents[0]}")
#     logger.info(f"First items: label {labels[0]}")
#
#     # wrapper = ActorWrapper()
#     # results = wrapper.process(sents, labels)
#     with open("actor_combined_split_dev.bin", "rb") as file:
#         results = pickle.load(file)
#     logger.info(f"First result: {results[0]}")
#
#     pred_entities, true_entities = get_series_bio(results)
#     print(classification_report(true_entities, pred_entities))
#
#     preds, trues = get_list_bio(results)
#     print(token_classification_report(trues, preds))
#
#     new_pred_entities = list()
#     for sent, result in zip(sents, results):
#         res = get_rule_fixes(sent, result)
#         new_pred_entities.append(res)
#
#     pred_entities, true_entities = get_series_bio(new_pred_entities)
#     print(classification_report(true_entities, pred_entities))
#
#     preds, trues = get_list_bio(new_pred_entities)
#     print(token_classification_report(trues, preds))
#
#
# #        precision    recall  f1-score   support
# #
# #     Actor    0.91018   0.89676   0.90342       339
# #
# #
# #        precision    recall  f1-score   support
# #
# # Actor    0.89218   0.96415   0.92677       753
# #
# #            precision    recall  f1-score   support
# #
# #     Actor    0.91018   0.89676   0.90342       339
# #
# #
# #        precision    recall  f1-score   support
# #
# # Actor    0.89218   0.96415   0.92677       753
#
#
# def test_measure_bert_resource_rules_prec():
#     data = list(load_dataset("pretrained_data/2022/resource/split_dev.jsonl"))
#     sents = [d[1] for d in data]
#     labels = [d[2] for d in data]
#     logger.info(f"First items: sent {sents[0]}")
#     logger.info(f"First items: label {labels[0]}")
#
#     wrapper = ResourceWrapper()
#     results = wrapper.process(sents, labels)
#     logger.info(f"First result: {results[0]}")
#
#     pred_entities, true_entities = get_series_bio(results)
#     print(classification_report(true_entities, pred_entities))
#
#     new_pred_entities = list()
#     for sent, result in zip(sents, results):
#         res = get_rule_fixes(sent, result, resource_plugins)
#         new_pred_entities.append(res)
#
#     pred_entities, true_entities = get_series_bio(new_pred_entities)
#     print(classification_report(true_entities, pred_entities))
#
#
# #            precision    recall  f1-score   support
# #
# #  Resource    0.64066   0.68262   0.66098       397
# #
# # micro avg    0.64066   0.68262   0.66098       397
# # macro avg    0.64066   0.68262   0.66098       397
#
#
# def test_measure_bert_intention_verb_prec():
#     data = list(load_dataset("pretrained_data/2022/task/verb/split_dev.jsonl"))
#     sents = [d[1] for d in data]
#     labels = [d[2] for d in data]
#     logger.info(f"First items: sent {sents[0]}")
#     logger.info(f"First items: label {labels[0]}")
#
#     wrapper = IntentionWrapper()
#     results = wrapper.process(sents, labels)
#     logger.info(f"First result: {results[0]}")
#
#     pred_entities, true_entities = get_series_bio(results)
#
#     print(classification_report(true_entities, pred_entities))
#
#     #            precision    recall  f1-score   support
#     #
#     #       Aux    0.73494   0.85915   0.79221        71
#     #      Cond    0.90909   0.80000   0.85106        25
#     #      Core    0.92464   0.94100   0.93275       339
#
#
# def test_measure_bert_intention_rules_prec():
#     data = list(load_dataset("pretrained_data/2022/task/verb/split_dev.jsonl"))
#     sents = [d[1] for d in data]
#     labels = [d[2] for d in data]
#     logger.info(f"First items: sent {sents[0]}")
#     logger.info(f"First items: label {labels[0]}")
#
#     # wrapper = IntentionWrapper()
#     # results = wrapper.process(sents, labels)
#     with open("intention_split_dev.bin", "rb") as file:
#         results = pickle.load(file)
#     logger.info(f"First result: {results[0]}")
#
#     pred_entities, true_entities = get_series_bio(results)
#     print(classification_report(true_entities, pred_entities))
#
#     # preds, trues = get_list_bio(results)
#     # print(token_classification_report(trues, preds))
#
#     new_pred_entities = list()
#     for sent, result in zip(sents, results):
#         res = get_rule_fixes(sent, result, intention_plugins)
#         new_pred_entities.append(res)
#
#     pred_entities, true_entities = get_series_bio(new_pred_entities)
#     print(classification_report(true_entities, pred_entities))
#
#     # preds, trues = get_list_bio(new_pred_entities)
#     # print(token_classification_report(trues, preds))
