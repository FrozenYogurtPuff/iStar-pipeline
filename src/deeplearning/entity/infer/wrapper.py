import logging
from abc import ABC
from typing import Any

from src.deeplearning.entity.infer.actor import InferActor, InferCombinedActor
from src.deeplearning.entity.infer.base import InferBase
from src.deeplearning.entity.infer.intention import InferIntention
from src.deeplearning.entity.infer.resource import InferResource
from src.deeplearning.entity.infer.result import BertResult
from src.utils.spacy import get_bio_sent_from_char_spans

logger = logging.getLogger(__name__)


class Wrapper(ABC):
    def __init__(self):
        self.inferrer: InferBase = NotImplemented

    @staticmethod
    def make_dict(
        sent: str, tokens: list[str], labels: list[str] | None
    ) -> dict[str, Any]:
        return {"sent": sent, "words": tokens, "labels": labels}

    @staticmethod
    def prepare(
        sents: str | list[str], labels: list | list[list] | None
    ) -> list[dict[str, Any]]:
        data = list()
        if not isinstance(sents, list):
            sents = [sents]
            labels = [labels]
        if not labels:
            labels = [None] * len(sents)

        for sent, label in zip(sents, labels):
            tokens, label = get_bio_sent_from_char_spans(sent, label)
            data.append(Wrapper.make_dict(sent, tokens, label))

        return data

    def infer(self, data: list[dict[str, Any]]) -> list[BertResult]:
        result = self.inferrer.predict(data)

        ret = list()
        preds_list, trues_list, matrix, tokens_bert, labs = result
        assert (
            len(preds_list)
            == len(trues_list)
            == len(matrix)
            == len(tokens_bert)
        )
        for p, ts, m, tk in zip(preds_list, trues_list, matrix, tokens_bert):
            ret.append(BertResult(p, ts, m, tk, labs))
        return ret

    def process(
        self, sents: str | list[str], labels: list | list[list] | None = None
    ) -> list[BertResult]:
        data = self.prepare(sents, labels)
        return self.infer(data)


class ActorWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.inferrer = InferActor()


class ActorCombinedWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.inferrer = InferCombinedActor()


class ResourceWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.inferrer = InferResource()


class IntentionWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.inferrer = InferIntention()
