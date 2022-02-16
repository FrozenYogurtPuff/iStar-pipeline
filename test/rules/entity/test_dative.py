import logging
import spacy
import spacy_alignments as tokenizations

import sys
print(sys.path)

from src.deeplearning import wrap_entity_oneline
from src.rules.entity.dative import dative


def test_dative():
    nlp: spacy.language.Language = spacy.load('en_core_web_lg')
    sent = "Show things to Anna."

    s = nlp(sent)
    spacy_tokens = [i.text for i in s]
    b, _, _, bert_tokens = wrap_entity_oneline(sent)
    assert len(b) != 1
    assert len(bert_tokens) != 1

    s2b, _ = tokenizations.get_alignments(spacy_tokens, bert_tokens)
    result = dative(s[:], b, s2b)
    logging.getLogger(__name__).debug(result)


if __name__ == '__main__':
    test_dative()
