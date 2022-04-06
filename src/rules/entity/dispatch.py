from __future__ import annotations

import logging

import spacy_alignments as tokenizations

from src.deeplearning.infer.result import BertResult
from src.rules.dispatch import dispatch
from src.utils.spacy import get_spacy
from src.utils.typing import RulePlugins

logger = logging.getLogger(__name__)


def get_rule_fixes(
    sent: str, b: BertResult, funcs: RulePlugins | None = None
) -> BertResult:
    nlp = get_spacy()
    logger.info(sent)
    s = nlp(sent)[:]
    spacy_tokens = [i.text for i in s]
    s2b, _ = tokenizations.get_alignments(spacy_tokens, b.tokens)
    result = dispatch(s, b, s2b, funcs=funcs) if funcs else dispatch(s, b, s2b)
    logger.debug(result)
    return b.apply_fix(result)
