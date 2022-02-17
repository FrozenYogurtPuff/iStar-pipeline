import spacy
import logging
import spacy_alignments as tokenizations

from src.rules.entity.dispatch import dispatch
from src.deeplearning import wrap_entity_oneline


def test_dispatch():
    nlp: spacy.language.Language = spacy.load('en_core_web_lg')
    sents = ["Bought me these books.", "Show things to Anna."]
    for sent in sents:
        s = nlp(sent)
        spacy_tokens = [i.text for i in s]
        preds_list, trues_list, matrix, bert_tokens = wrap_entity_oneline(sent)
        assert len(preds_list) != 1
        assert len(bert_tokens) != 1
        s2b, _ = tokenizations.get_alignments(spacy_tokens, bert_tokens)
        result = dispatch(s[:], preds_list, s2b)
        logging.getLogger(__name__).warning(result)


if __name__ == '__main__':
    test_dispatch()
