#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:44:17 2019

@author: weetee
"""

import logging
import os
import pickle

import torch
from tqdm import tqdm

tqdm.pandas(desc="prog-bar")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")


def load_pickle(filename):
    completeName = os.path.join("./data/", filename)
    with open(completeName, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data


class infer_from_trained(object):
    def __init__(self, args=None):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()

        self.entities_of_interest = [
            "PERSON",
            "NORP",
            "FAC",
            "ORG",
            "GPE",
            "LOC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
            "PER",
        ]

        logger.info("Loading tokenizer and model...")
        from .train_funcs import load_state

        if self.args.model_no == 0:
            from ..model.BERT.modeling_bert import BertModel as Model

            model = args.model_size  #'bert-base-uncased'
            lower_case = True
            model_name = "BERT"
            self.net = Model.from_pretrained(
                model,
                force_download=False,
                model_size=args.model_size,
                task="classification",
                n_classes_=self.args.num_classes,
            )
        elif self.args.model_no == 1:
            from ..model.ALBERT.modeling_albert import AlbertModel as Model

            model = args.model_size  #'albert-base-v2'
            lower_case = False
            model_name = "ALBERT"
            self.net = Model.from_pretrained(
                model,
                force_download=False,
                model_size=args.model_size,
                task="classification",
                n_classes_=self.args.num_classes,
            )
        elif args.model_no == 2:  # BioBert
            from ..model.BERT.modeling_bert import BertConfig, BertModel

            model = "bert-base-uncased"
            lower_case = False
            model_name = "BioBERT"
            config = BertConfig.from_pretrained(
                "./additional_models/biobert_v1.1_pubmed/bert_config.json"
            )
            self.net = BertModel.from_pretrained(
                pretrained_model_name_or_path="./additional_models/biobert_v1.1_pubmed/biobert_v1.1_pubmed.bin",
                config=config,
                force_download=False,
                model_size="bert-base-uncased",
                task="classification",
                n_classes_=self.args.num_classes,
            )

        self.tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        self.net.resize_token_embeddings(len(self.tokenizer))
        if self.cuda:
            self.net.cuda()
        start_epoch, best_pred, amp_checkpoint = load_state(
            self.net, None, None, self.args, load_best=False
        )
        logger.info("Done!")

        self.e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        self.pad_id = self.tokenizer.pad_token_id
        self.rm = load_pickle("relations.pkl")

    def get_e1e2_start(self, x):
        e1_e2_start = (
            [i for i, e in enumerate(x) if e == self.e1_id][0],
            [i for i, e in enumerate(x) if e == self.e2_id][0],
        )
        return e1_e2_start

    def infer_one_sentence(self, sentence):
        self.net.eval()
        tokenized = self.tokenizer.encode(sentence)
        # print(tokenized)
        e1_e2_start = self.get_e1e2_start(tokenized)
        # print(e1_e2_start)
        tokenized = torch.LongTensor(tokenized).unsqueeze(0)
        e1_e2_start = torch.LongTensor(e1_e2_start).unsqueeze(0)
        attention_mask = (tokenized != self.pad_id).float()
        token_type_ids = torch.zeros(
            (tokenized.shape[0], tokenized.shape[1])
        ).long()

        if self.cuda:
            tokenized = tokenized.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

        with torch.no_grad():
            classification_logits = self.net(
                tokenized,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                Q=None,
                e1_e2_start=e1_e2_start,
            )
            predicted = (
                torch.softmax(classification_logits, dim=1).max(1)[1].item()
            )
        print("Sentence: ", sentence)
        print("Predicted: ", self.rm.idx2rel[predicted].strip(), "\n")
        return predicted

    def infer_sentence(self, sentence):
        return self.infer_one_sentence(sentence)
