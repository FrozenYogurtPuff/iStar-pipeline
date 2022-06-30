# Logger
import logging

from src.deeplearning.entity.infer.wrapper import (
    ActorWrapper,
    IntentionWrapper,
)

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        self.sents: list[str] = list()
        self.results: list[dict] = list()
        self.actor = ActorWrapper()
        self.intention = IntentionWrapper()

    @staticmethod
    def make_dict() -> dict:
        raise NotImplemented  # TODO

    def process(self):
        actors = self.actor.process(self.sents)
        intentions = self.intention.process(self.sents)
        entities = list(zip(actors, intentions))
        for sent, entity in zip(self.sents, entities):
            actor, intention = entity
            # Do some magic interacts here
