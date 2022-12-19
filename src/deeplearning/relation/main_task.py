#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""
import itertools
import logging
import pickle
from argparse import ArgumentParser

from src.deeplearning.relation import kfold
from src.deeplearning.relation.code.tasks.infer import infer_from_trained

"""
This fine-tunes the BERT model on SemEval, FewRel tasks
"""

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")


parser = ArgumentParser()
parser.add_argument(
    "--task", type=str, default="istar", help="semeval, fewrel"
)
parser.add_argument(
    "--train_data",
    type=str,
    default=f"./pretrained_data/2022_Kfold/relation/admin.jsonl",
    help="training data .txt file path",
)
parser.add_argument(
    "--use_pretrained_blanks",
    type=int,
    default=0,
    help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model",
)
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
parser.add_argument("--num_epochs", type=int, default=40, help="No of epochs")
parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
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
parser.add_argument(
    "--train", type=int, default=0, help="0: Don't train, 1: train"
)
parser.add_argument(
    "--infer", type=int, default=0, help="0: Don't infer, 1: Infer"
)

args = parser.parse_args()


def proceed_sentence(text: str, entities: list):
    ret = list()
    for en1, en2 in itertools.combinations(entities, 2):
        s1, e1 = en1["startOffset"], en1["endOffset"]
        s2, e2 = en2["startOffset"], en2["endOffset"]
        sentence = (
            text[:s1]
            + "[E1]"
            + text[s1:e1]
            + "[/E1]"
            + text[e1:s2]
            + "[E2]"
            + text[s2:e2]
            + "[/E2]"
            + text[e2:]
        )
        print(sentence)
        infer = infer_from_trained(args, detect_entities=True)
        result = infer.infer_sentence(sentence)
        print(result)
        if result == 1:
            continue
        label_id = -1
        if result == 0:  # dependency
            label_id = 11
        elif result == 2:  # isa
            label_id = 12
        elif result == 3:  # part-of
            label_id = 13
        else:
            raise "Illegal label"
        ret.append((en1["id"], en2["id"], label_id))
    print(ret)
    return ret


if __name__ == "__main__":
    K = 10

    # if (args.train == 1) and (args.task != "fewrel"):
    #     net = train_and_fit(args)
    #
    # if (args.infer == 1) and (args.task != "fewrel"):
    #     # 从 pickle 加载所需模型配置
    #     inferer = infer_from_trained(args, detect_entities=True)
    #
    #     test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
    #     inferer.infer_sentence(test, detect_entities=False)
    #     test2 = "After eating the chicken, he developed a sore throat the next morning."
    #     inferer.infer_sentence(test2, detect_entities=True)
    #
    #     while True:
    #         sent = input(
    #             "Type input sentence ('quit' or 'exit' to terminate):\n"
    #         )
    #         if sent.lower() in ["quit", "exit"]:
    #             break
    #         inferer.infer_sentence(sent, detect_entities=False)
    #
    # if args.task == "fewrel":
    #     fewrel = FewRel(args)
    #     meta_input, e1_e2_start, meta_labels, outputs = fewrel.evaluate()

    # 评估
    p_all, r_all, f1_all = list(), list(), list()
    for i in range(K):
        kfold.select = i
        # net = train_and_fit(args)
        inferer = infer_from_trained(args, detect_entities=False)
        tp, fp, tn, fn = 0, 0, 0, 0
        with open(
            f"pretrained_data/2022_Kfold/relation/{i}/df_test.pkl", "rb"
        ) as pkl_file:
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
            print(i, p, r, f1)

        p_all.append(p)
        r_all.append(r)
        f1_all.append(f1)

    assert len(p_all) == len(r_all) == len(f1_all) == K
    p_avg = sum(p_all) / len(p_all)
    r_avg = sum(r_all) / len(r_all)
    f1_avg = sum(f1_all) / len(f1_all)
    print(str(p_avg), str(r_avg), str(f1_avg))
