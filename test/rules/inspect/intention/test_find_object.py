import logging
from test.rules.utils.load_dataset import load_dataset

from spacy_alignments import tokenizations

from src.deeplearning.entity.infer.wrapper import IntentionWrapper
from src.rules.intention.find_object import find_object
from src.utils.spacy import char_idx_to_word, get_spacy

logger = logging.getLogger(__name__)


def label_split(labels: list) -> tuple[list, list]:
    cores, nouns = list(), list()
    for label in labels:
        if label[2] == "Core":
            cores.append(label)
        elif label[2] == "Noun":
            nouns.append(label)
        else:
            raise Exception(f"Illegal label {label[2]}")
    return cores, nouns


def metrics(pred: set, target: set) -> tuple[int, int, int]:
    tp = len(target.intersection(pred))
    fn = len(target - pred)
    fp = len(pred - target)
    return tp, fn, fp


# Precision: 0.9184952978056427
# Recall: 0.8960244648318043
def test_find_object_prec():
    data = list(
        load_dataset(
            "pretrained_data/2022/task/verb_noun_dev/dev_noun_admin.jsonl"
        )
    )
    nlp = get_spacy()
    tp, fn, fp = 0, 0, 0
    for item in data:
        text = item[1]
        anno = item[2]
        cs, ns = label_split(anno)
        predicts, grounds = set(), set()
        sent = nlp(text)
        for c in cs:
            key = char_idx_to_word(sent, c[0], c[1])
            predicts.update(find_object(key))
        for n in ns:
            key = char_idx_to_word(sent, n[0], n[1])
            grounds.add(key.root)
        logger.debug(sent)
        logger.debug(predicts)
        logger.debug(grounds)
        if len(predicts) != len(grounds.intersection(predicts)):
            cores_text = list()
            for c in cs:
                key = char_idx_to_word(sent, c[0], c[1])
                cores_text.append(key)
            logger.warning(cores_text)
        metric = metrics(predicts, grounds)
        tp += metric[0]
        fn += metric[1]
        fp += metric[2]
    logger.info(f"Precision: {tp / (tp + fp)}")
    logger.info(f"Recall: {tp / (tp + fn)}")


# Precision: 0.8979289940828402
# Recall: 0.912781954887218
def test_bert_verb_with_object_prec():
    data = list(
        load_dataset(
            "pretrained_data/2022/task/verb_noun_dev/dev_noun_admin.jsonl"
        )
    )
    nlp = get_spacy()
    sents = [d[1] for d in data]
    core_labels, noun_labels = list(), list()
    for item in data:
        labs = item[2]
        cs, ns = label_split(labs)
        core_labels.append(cs)
        noun_labels.append(ns)
    wrapper = IntentionWrapper()
    results = wrapper.process(sents, core_labels)
    tp, fn, fp = 0, 0, 0
    for i, result in enumerate(results):
        sent = nlp(sents[i])
        grounds = set()
        for n in noun_labels[i]:
            key = char_idx_to_word(sent, n[0], n[1])
            grounds.add(key.root)
        for c in core_labels[i]:
            key = char_idx_to_word(sent, c[0], c[1])
            grounds.add(key.root)
        spacy_tokens = [i.text for i in sent]
        s2b, b2s = tokenizations.get_alignments(spacy_tokens, result.tokens)
        predicts = set()
        for idx, lab in enumerate(result.preds):
            if lab.endswith("Core"):
                sidx = b2s[idx]
                if not sidx:
                    continue
                span = sent[sidx[0] : sidx[-1] + 1]
                predicts.add(span.root)
                predicts.update(find_object(span))
        metric = metrics(predicts, grounds)
        tp += metric[0]
        fn += metric[1]
        fp += metric[2]
    logger.info(f"Precision: {tp / (tp + fp)}")
    logger.info(f"Recall: {tp / (tp + fn)}")
