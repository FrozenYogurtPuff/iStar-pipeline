from __future__ import annotations

import logging

import spacy_alignments as tokenizations

from src.deeplearning.entity.infer.result import BertResult
from src.rules.dispatch import dispatch
from src.rules.intention.aux_slice.dispatch import dispatch as dispatch_slice
from src.utils.spacy_utils import get_spacy
from src.utils.typing import RulePlugins

logger = logging.getLogger(__name__)


def get_rule_fixes(
    sent: str,
    b: BertResult,
    funcs: RulePlugins | None = None,
    is_slice: bool = True,
) -> BertResult:
    nlp = get_spacy()
    logger.info(sent)
    s = nlp(sent)[:]
    spacy_tokens = [i.text for i in s]
    s2b, _ = tokenizations.get_alignments(spacy_tokens, b.tokens)
    result = dispatch(s, b, s2b, funcs=funcs) if funcs else dispatch(s, b, s2b)
    fix_result = b.apply_fix(result)
    if is_slice:
        slices = dispatch_slice(s)
        slice_result = fix_result.apply_slices(slices)
        logger.debug(slice_result)
        return slice_result
    else:
        logger.debug(fix_result)
        return fix_result
