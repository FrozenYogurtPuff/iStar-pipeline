#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""
import logging
from argparse import ArgumentParser

from tasks.infer import infer_from_trained

"""
This fine-tunes the BERT model on SemEval, FewRel tasks
"""

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--num_classes", type=int, default=4, help="number of relation classes"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--gradient_acc_steps",
        type=int,
        default=1,
        help="No. of steps of gradient accumulation",
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipped gradient norm"
    )
    parser.add_argument(
        "--fp16",
        type=int,
        default=0,
        help="1: use mixed precision ; 0: use floating point 32",
    )  # mixed precision doesn't seem to train well
    parser.add_argument(
        "--num_epochs", type=int, default=25, help="No of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.00007, help="learning rate"
    )
    parser.add_argument(
        "--model_no",
        type=int,
        default=0,
        help="""Model ID: 0 - BERT\n
                                                                            1 - ALBERT\n
                                                                            2 - BioBERT""",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="bert-base-uncased",
        help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For ALBERT: 'albert-base-v2',\
                                                                                                'albert-large-v2'\
                                                                                    For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)",
    )

    args = parser.parse_args()

    # 从 pickle 加载所需模型配置
    inferer = infer_from_trained(args)
    test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
    inferer.infer_sentence(test)
