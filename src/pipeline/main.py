# spaCy
# Logger
import logging
import spacy

from src.utils.spacy import get_spacy


class Pipeline:
    def __init__(self):
        self.nlp: spacy.language.Language = get_spacy()
        self._logger: logging.Logger = logging.getLogger(__name__)

    def run(self):
        pass
