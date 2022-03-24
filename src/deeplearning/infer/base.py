# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

import argparse
import logging
import os
import random
import re
from abc import ABC

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from src.deeplearning.models.model_ner import (
    MODEL_FOR_SOFTMAX_NER_MAPPING,
    MODEL_PRETRAINED_CONFIG_ARCHIVE_MAPPING,
    AutoModelForSoftmaxNer,
)
from src.deeplearning.utils.utils_ner import (
    InputExample,
    convert_examples_to_features,
    get_labels,
    read_examples_from_file,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SOFTMAX_NER_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_MAPS = tuple
ALL_MODELS = sum(
    (
        tuple(MODEL_PRETRAINED_CONFIG_ARCHIVE_MAPPING[conf].keys())
        for conf in MODEL_CONFIG_CLASSES
    ),
    (),
)
TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

logger = logging.getLogger(__name__)


class InferBase(ABC):
    def __init__(
        self,
        data_dir,
        model_type,
        model_name_or_path,
        output_dir,
        label,
    ):
        parser = argparse.ArgumentParser()
        # Required parameters
        parser.add_argument(
            "--data_dir",
            default=data_dir,
            type=str,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )
        parser.add_argument(
            "--model_type",
            default=model_type,
            type=str,
            help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
        )
        parser.add_argument(
            "--model_name_or_path",
            default=model_name_or_path,
            type=str,
            help="Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(ALL_MODELS),
        )
        parser.add_argument(
            "--output_dir",
            default=output_dir,
            type=str,
            help="The output directory where the model predictions and checkpoints will be written.",
        )

        # Other parameters
        parser.add_argument(
            "--labels",
            default=label,
            type=str,
            help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
        )
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--do_lower_case",
            action="store_true",
            help="Set this flag if you are using an uncased model.",
        )
        parser.add_argument(
            "--keep_accents",
            action="store_const",
            const=True,
            help="Set this flag if model is trained with accents.",
        )
        parser.add_argument(
            "--strip_accents",
            action="store_const",
            const=True,
            help="Set this flag if model is trained without accents.",
        )
        parser.add_argument(
            "--use_fast",
            action="store_const",
            const=True,
            help="Set this flag to use fast tokenization.",
        )
        parser.add_argument(
            "--per_gpu_eval_batch_size",
            default=16,
            type=int,
            help="Batch size per GPU/CPU for evaluation.",
        )
        parser.add_argument(
            "--loss_type",
            default="lsr",
            type=str,
            help="The loss function to optimize.",
        )
        parser.add_argument(
            "--no_cuda",
            action="store_true",
            help="Avoid using CUDA when available",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="random seed for initialization",
        )

        parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="For distributed training: local_rank",
        )
        self.args, _ = parser.parse_known_args()

        # Setup CUDA, GPU & distributed training
        if self.args.local_rank == -1 or self.args.no_cuda:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available() and not self.args.no_cuda
                else "cpu"
            )
            self.args.n_gpu = (
                0 if self.args.no_cuda else torch.cuda.device_count()
            )
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.args.n_gpu = 1
        self.args.device = device

        logger.info(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
            self.args.local_rank,
            self.args.device,
            self.args.n_gpu,
            bool(self.args.local_rank != -1),
        )

        # Set seed
        self.set_seed()

        # Prepare CONLL-2003 task
        self.labels = get_labels(self.args.labels)
        self.num_labels = len(self.labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        # Load pretrained model and tokenizer
        if self.args.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        self.args.model_type = self.args.model_type.lower()
        self.config = AutoConfig.from_pretrained(
            self.args.config_name
            if self.args.config_name
            else self.args.model_name_or_path,
            num_labels=self.num_labels,
            id2label={str(i): label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)},
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        #####
        setattr(self.config, "loss_type", self.args.loss_type)
        #####
        tokenizer_args = {
            k: v
            for k, v in vars(self.args).items()
            if v is not None and k in TOKENIZER_ARGS
        }

        logger.info("Tokenizer arguments: %s", tokenizer_args)

        if self.args.local_rank == 0:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        logger.info("Training/evaluation parameters %s", self.args)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.output_dir, local_files_only=True, **tokenizer_args
        )
        checkpoint = os.path.join(self.args.output_dir, "best_checkpoint")
        self.model = AutoModelForSoftmaxNer.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.model.to(self.args.device)

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

    def evaluate(self, data, prefix=""):
        eval_dataset = self.load_and_cache_examples(data)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(
            1, self.args.n_gpu
        )
        # Note that DistributedSampler samples randomly
        eval_sampler = (
            SequentialSampler(eval_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
        )

        # multi-gpu evaluate
        if self.args.n_gpu > 1 and not isinstance(
            self.model, torch.nn.DataParallel
        ):
            self.model = torch.nn.DataParallel(self.model)

        # TODO: tokenizers warning being forked, Disabling parallelism to avoid deadlocks
        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        trues = None
        input_ids = None
        valid_mask = None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "valid_mask": batch[2],
                    "labels": batch[4],
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2]
                        if self.args.model_type in ["bert", "xlnet"]
                        else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                if self.args.n_gpu > 1:
                    tmp_eval_loss = (
                        tmp_eval_loss.mean()
                    )  # mean() to average on multi-gpu parallel evaluating
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                trues = inputs["labels"].detach().cpu().numpy()
                input_ids = inputs["input_ids"].detach().cpu().numpy()
                valid_mask = inputs["valid_mask"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                trues = np.append(
                    trues, inputs["labels"].detach().cpu().numpy(), axis=0
                )
                input_ids = np.append(
                    input_ids,
                    inputs["input_ids"].detach().cpu().numpy(),
                    axis=0,
                )
                valid_mask = np.append(
                    valid_mask,
                    inputs["valid_mask"].detach().cpu().numpy(),
                    axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps
        preds_argmax = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.labels)}

        preds_list = [[] for _ in range(preds.shape[0])]
        trues_list = [[] for _ in range(trues.shape[0])]
        matrix = [[] for _ in range(preds.shape[0])]
        tokens_bert = list()

        for i in range(trues.shape[0]):
            tokens = list()
            for j in range(trues.shape[1]):
                if valid_mask[i, j] != 0 and input_ids[i, j] not in [101, 102]:
                    tokens.append(
                        self.tokenizer.convert_ids_to_tokens(
                            int(input_ids[i, j])
                        )
                    )
                if trues[i, j] != self.pad_token_label_id:
                    res = softmax(torch.from_numpy(preds[i, j, :]), dim=0)
                    matrix[i].append(res.tolist())
                    preds_list[i].append(label_map[preds_argmax[i][j]])
                    trues_list[i].append(label_map[trues[i][j]])
            assert len(tokens) == len(matrix[i])
            tokens_bert.append(tokens)

        return preds_list, trues_list, matrix, tokens_bert, self.labels

    @classmethod
    def read_examples_from_json(cls, data):
        """
        Send sents via json format

        :param data: [{sent: str, labels: list[str]]} | None
        :return: list[InputExample]
        """
        guid_index = 1
        examples = []
        for item in data:
            if "words" not in item:
                item["words"] = list(
                    filter(str.split, re.split('([,.?!":()/ ])', item["sent"]))
                )
            if "labels" not in item or item["labels"] is None:
                item["labels"] = ["O" for _ in range(len(item["words"]))]
            examples.append(
                InputExample(
                    guid="{}".format(guid_index),
                    words=item["words"],
                    labels=item["labels"],
                )
            )
            guid_index += 1
        return examples

    # 制作数据集
    def load_and_cache_examples(self, data):
        logger.info(
            "Creating features from dataset file at %s", self.args.data_dir
        )
        # examples = read_examples_from_file(args.data_dir, mode)
        if data:
            examples = self.read_examples_from_json(data)
        else:
            examples = read_examples_from_file(self.args.data_dir, "dev")
        features = convert_examples_to_features(
            examples,
            self.labels,
            self.args.max_seq_length,
            self.tokenizer,
            cls_token_at_end=bool(self.args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.args.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=bool(self.args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(self.args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_valid_mask = torch.tensor(
            [f.valid_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )

        dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_valid_mask,
            all_segment_ids,
            all_label_ids,
        )
        return dataset

    def predict(self, data):
        return self.evaluate(data, prefix="test")

    @property
    def model_wrapped(self):
        return NotImplemented
