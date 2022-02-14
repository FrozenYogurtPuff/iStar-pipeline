# 目的：让原生的evaluate方法支持读入conll，并返回ground truth、推理结果、推测矩阵、对应文本信息
# * 对应文本信息
from src.infer.infer_task import predict_task as bert_task
from metric import classification_report
from src.utils.utils_metrics import get_entities_bio


def get_bert_task_conll():
    return bert_task(data=None)


bert_task_pred_list, bert_task_true_list, bert_task_matrix, bert_task_tokens = get_bert_task_conll()
print(len(bert_task_pred_list))
print(len(bert_task_true_list))
print(len(bert_task_matrix))
print(len(bert_task_tokens))


true_entities = get_entities_bio(bert_task_true_list)
pred_entities = get_entities_bio(bert_task_pred_list)

print(classification_report(true_entities, pred_entities))
