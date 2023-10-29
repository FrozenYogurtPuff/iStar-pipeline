# The function structure has changed!

# import logging
# import spacy
# import spacy_alignments as tokenizations
#
# from src.deeplearning.entity import wrap_entity_oneline
# from src.rules.actor.dative_ADP import dative_ADP

# def test_dative():
#     nlp: spacy.language.Language = spacy.load('en_core_web_trf')
#     sent = "Show things to Anna."
#
#     s = nlp(sent)
#     spacy_tokens = [i.text for i in s]
#     preds_list, trues_list, matrix, bert_tokens = wrap_entity_oneline(sent)
#     assert len(preds_list) != 1
#     assert len(bert_tokens) != 1
#
#     s2b, _ = tokenizations.get_alignments(spacy_tokens, bert_tokens)
#     result = dative_ADP(s[:], preds_list, s2b)
#     logging.getLogger(__name__).debug(result)
#
#
# if __name__ == '__main__':
#     test_dative()
