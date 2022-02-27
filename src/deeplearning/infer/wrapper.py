from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import src.deeplearning.infer.result as br
from src.deeplearning.infer.entity import get_entity_model
from src.deeplearning.infer.intention import get_intention_model
from src.deeplearning.infer.utils import label_mapping_bio
from src.utils.spacy import char_idx_to_word_idx, get_spacy
from src.utils.typing import BertUnionLabelBio, DatasetUnionLabel

logger = logging.getLogger(__name__)


def set_label(
    ret: List[BertUnionLabelBio],
    start: int,
    end: int,
    types: Tuple[BertUnionLabelBio, BertUnionLabelBio],
):
    ret[start] = types[0]
    for i in range(start + 1, end):
        ret[i] = types[1]


# [(0, 10, "Actor"), (25, 64, "Resource")] -> ['O', ..., 'Actor', 'Resource']
def handle_input(
    ss: str, labels: Optional[List[DatasetUnionLabel]] = None
) -> Tuple[List[str], List[BertUnionLabelBio]]:
    nlp = get_spacy()
    sent = nlp(ss)

    o: BertUnionLabelBio = "O"
    ret = [o] * len(sent)
    if labels:
        for s, e, label in labels:
            sw, ew = char_idx_to_word_idx(sent[:], s, e)
            set_label(ret, sw, ew, label_mapping_bio(label))

    sent = [s.text for s in sent]
    return sent, ret


def infer_wrapper(
    ident: Literal["Entity", "Intention"],
    sents: Union[str, List[str]],
    labels: Optional[Union[List[Any], List[List[Any]]]] = None,
) -> List[br.BertResult]:
    def make_dict(
        s: str, w: List[str], la: Optional[List[BertUnionLabelBio]] = None
    ) -> Dict[str, Any]:
        return {"sent": s, "words": w, "labels": la}

    if ident not in ["Entity", "Intention"]:
        logger.error(f"Unexcepted identifier {ident}")
        raise Exception("Illegal identifier pattern")
    logger.info(f"Infer type: {ident}")

    data = list()
    if isinstance(sents, list):
        logger.info(f"Infer pattern: list with length {len(sents)}")
        if labels:
            for sent, label in zip(sents, labels):
                label = [tuple(lab) for lab in label]
                tokens, label = handle_input(sent, label)
                data.append(make_dict(sent, tokens, label))
        else:
            for sent in sents:
                tokens, label = handle_input(sent)
                data.append(make_dict(sent, tokens, label))
    else:
        logger.info("Infer pattern: single string")
        label = [tuple(lab) for lab in labels] if labels else None
        tokens, label = handle_input(sents, label)
        data.append(make_dict(sents, tokens, label))

    if ident == "Entity":
        result = get_entity_model().predict(data)
    else:
        result = get_intention_model().predict(data)

    ret: List[br.BertResult] = list()
    preds_list, trues_list, matrix, tokens_bert, labs = result
    assert (
        len(preds_list) == len(trues_list) == len(matrix) == len(tokens_bert)
    )
    for p, ts, m, tk in zip(preds_list, trues_list, matrix, tokens_bert):
        ret.append(br.BertResult(p, ts, m, tk, labs))
    return ret
