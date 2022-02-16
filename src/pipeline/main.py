# spaCy
import spacy

# Logger
import logging


class Pipeline:
    def __init__(self):
        self.nlp: spacy.language.Language = spacy.load('en_core_web_lg')
        self._logger: logging.Logger = logging.getLogger(__name__)

    def run(self):
        pass
