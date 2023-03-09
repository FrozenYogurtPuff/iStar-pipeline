from __future__ import annotations

import logging
from typing import Callable

import spacy_alignments as tokenizations

from src.deeplearning.entity.infer.result import BertResult
from src.rules.config import actor_plugins
from src.rules.dispatch import dispatch, prob_bert_merge
from src.utils.spacy_utils import get_spacy
from src.utils.typing import RulePlugins

logger = logging.getLogger(__name__)


def get_rule_fixes(
    sent: str,
    b: BertResult,
    funcs: RulePlugins | None = None,
    bert_func: Callable = None,
) -> BertResult:
    nlp = get_spacy()
    logger.info(sent)
    s = nlp(sent)[:]
    spacy_tokens = [i.text for i in s]
    s2b, _ = tokenizations.get_alignments(spacy_tokens, b.tokens)
    funcs = actor_plugins if funcs is None else funcs
    bert_func = bert_func if bert_func else prob_bert_merge

    result = dispatch(s, b, s2b, funcs=funcs, bert_func=bert_func)
    logger.debug(result)
    return b.apply_fix(result)
