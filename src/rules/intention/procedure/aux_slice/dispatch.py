import logging
from typing import List, Optional

from src.rules.config import intention_aux_slice_plugins
from src.utils.spacy import get_token_idx, include_elem, token_not_end
from src.utils.typing import (
    FixIntentionLabel,
    HybridToken,
    IntentionRuleAuxPlugins,
    IntentionSlice,
    SpacySpan,
)

logger = logging.getLogger(__name__)


def need_slice(s: IntentionSlice, bidx: int, eidx: int) -> bool:
    if bidx == s[0]:
        return False
    if eidx == s[1]:
        return False
    if bidx > s[1]:
        return False
    if eidx < s[0]:
        return False
    return True


def dispatch(
    s: SpacySpan,
    seq_slices: Optional[List[IntentionSlice]] = None,
    funcs: IntentionRuleAuxPlugins = intention_aux_slice_plugins,
) -> List[IntentionSlice]:
    core: FixIntentionLabel = "Core"
    aux: FixIntentionLabel = "Aux"

    if seq_slices is None:
        seq_idx = get_token_idx(s)
        seq_s, seq_e = seq_idx[0], seq_idx[-1]
        seq_slices = [(seq_s, seq_e, core)]
    assert seq_slices is not None

    pool: List[HybridToken] = list()
    for func in funcs:
        pool.extend(func(s))
    pool = list(set(pool))

    # Chunk mapping
    for idx, p in enumerate(pool):
        for nc in s.noun_chunks:
            if include_elem(p, nc):
                pool[idx] = nc
                break

    while pool:
        p = pool.pop()
        idxes = get_token_idx(p)
        bidx, eidx = idxes[0], idxes[-1]
        for i, sli in enumerate(seq_slices):
            assert isinstance(i, int)
            assert sli[2] in [core, aux]
            if need_slice(sli, bidx, eidx):
                slice_type = core, aux
                for tok in s:
                    if tok.dep_ == "nsubj":
                        root_idx = get_token_idx(tok.head)[0]
                        if root_idx > eidx:
                            slice_type = aux, core
                        if token_not_end(tok) and tok.nbor(1).lower_ in [
                            "is",
                            "was",
                            "are",
                            "were",
                        ]:
                            slice_type = core, core
                        break
                if sli[2] == aux:
                    slice_type = aux, aux

                seq_slices[i] = (sli[0], eidx, slice_type[0])
                seq_slices.insert(i + 1, (bidx, sli[1], slice_type[1]))

    return seq_slices


# advcl 及其 head 不是 Aux
