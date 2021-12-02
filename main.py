import re

from infer_task import predict_task
from infer_entity import predict_entity

sent = 'Teacher Y asks student A to fill out a loan form and write down the following: information ' \
       'about the teacher in the classroom, the reason for borrowing the classroom, and the time ' \
       'for borrowing the classroom. '


def sent_tokenize(sent):
    return list(filter(str.split, re.split('([,|.|?|!|"|:|(|)|/| ])', sent)))


def get_bert_task(sent):
    data = [{'sent': sent}]
    return predict_task(data)


def get_bert_entity(sent):
    data = [{'sent': sent}]
    return predict_entity(data)


def output_pretty(tok, pred):
    for i in zip(tok, pred[0]):
        print(i)


pred = get_bert_entity(sent)
tok = sent_tokenize(sent)
output_pretty(tok, pred)

pred = get_bert_task(sent)
output_pretty(tok, pred)
