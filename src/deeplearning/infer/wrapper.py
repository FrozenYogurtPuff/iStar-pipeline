from src.deeplearning import predict_entity


def wrap_entity_oneline(sent, labels=None):
    data = {'sent': sent}
    if labels:
        data['labels'] = labels
    matrix, preds_list, trues_list, tokens_bert = predict_entity([data])
    matrix = matrix[0]
    preds_list = preds_list[0]
    trues_list = trues_list[0]
    tokens_bert = tokens_bert[0]
    return matrix, preds_list, trues_list, tokens_bert
