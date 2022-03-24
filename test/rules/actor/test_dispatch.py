import logging

import spacy

logger = logging.getLogger(__name__)


def test_dispatch():
    nlp: spacy.language.Language = spacy.load("en_core_web_lg")
    sents = [
        "Bought me these books.",
        "Show things to Anna.",
        "Carried out by immigrants",
    ]
    for sent in sents:
        s = nlp(sent)
        spacy_tokens = [i.text for i in s]
        # results = infer_wrapper("Entity", sent, None)  # TODO
        # assert len(results) == 1
        # result = results[0]
        # assert len(result.preds) != 1
        # assert len(result.tokens) != 1
        # s2b, _ = tokenizations.get_alignments(spacy_tokens, result.tokens)
        # res = dispatch(s[:], result, s2b)
        # logger.warning(res)


def test_dispatch_parallel():
    nlp: spacy.language.Language = spacy.load("en_core_web_lg")
    sents = [
        "Bought me these books.",
        "Show things to Anna.",
        "Carried out by immigrants",
    ]
    s = list(nlp.pipe(sents))
    spacy_tokens = [[i.text for i in ss] for ss in s]
    # results = infer_wrapper("Entity", sents, None)
    # for ss, spacy_token, result in zip(s, spacy_tokens, results):
    #     s2b, _ = tokenizations.get_alignments(spacy_token, result.tokens)
    #     res = dispatch(ss[:], result, s2b)
    #     logger.warning(res)


if __name__ == "__main__":
    test_dispatch()
