import logging

from spacy.tokens import Span, Token

from src.rules.config import intention_aux_slice_plugins
from src.utils.spacy import get_token_idx, include_elem, token_not_end
from src.utils.typing import IntentionRuleAuxPlugins, SeqSlicesTuple

logger = logging.getLogger(__name__)


def need_slice(s: SeqSlicesTuple, bidx: int, eidx: int) -> bool:
    if bidx == s.start:
        return False
    if eidx == s.end:
        return False
    if bidx > s.end:
        return False
    if eidx < s.start:
        return False
    return True


def dispatch(
    s: Span,
    seq_slices: list[SeqSlicesTuple] | None = None,
    slice_funcs: IntentionRuleAuxPlugins = intention_aux_slice_plugins,
) -> list[SeqSlicesTuple]:
    core: str = "Core"
    aux: str = "Aux"

    if seq_slices is None:
        seq_idx = get_token_idx(s)
        seq_s, seq_e = seq_idx[0], seq_idx[-1]
        seq_slices = [SeqSlicesTuple(seq_s, seq_e, core)]
    assert seq_slices is not None

    pool: list[Span | Token] = list()
    for func in slice_funcs:
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
            assert sli.type_ in [core, aux]
            if need_slice(sli, bidx, eidx):
                slice_type = core, aux
                for tok in s:
                    if tok.dep_ == "nsubj":
                        root_idx = get_token_idx(tok.head)[0]
                        if root_idx > eidx:
                            slice_type = aux, core
                        # `new is`
                        if token_not_end(tok) and tok.nbor(1).lower_ in [
                            "is",
                            "was",
                            "are",
                            "were",
                        ]:
                            slice_type = core, slice_type[-1]
                        break
                # `advcl`
                for tok in s:
                    if tok.dep_ == "advcl":
                        if get_token_idx(tok)[-1] < eidx:
                            # just change `[0]`
                            slice_type = core, slice_type[-1]
                        break
                if sli.type_ == aux:
                    slice_type = aux, aux

                seq_slices[i] = SeqSlicesTuple(sli[0], eidx, slice_type[0])
                seq_slices.insert(
                    i + 1, SeqSlicesTuple(bidx, sli[1], slice_type[1])
                )

    return seq_slices
