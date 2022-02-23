import collections.abc
import logging
from typing import Callable, Sequence, Tuple, Union

from src.utils.typing import SpacySpan, SpacyToken

logger = logging.getLogger(__name__)


# sent, 0, 14 -> 0, 2
def char_idx_to_word_idx(sent: SpacySpan, begin: int, end: int) -> Tuple[int, int]:
    biases = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    base_factor = 0
    strs = sent.char_span(begin, end)
    while strs is None:
        all_fail = True
        base_factor += 1
        for bias in biases:
            b, e = begin + bias[0] * base_factor, end + bias[1] * base_factor
            if not idx_valid(sent, (b, e), is_char=True):
                continue
            all_fail = False
            strs = sent.char_span(b, e)
            if strs is not None:
                if base_factor <= 1:
                    logger.warning(f'Sent: {sent.text}\n')
                    logger.warning(f'Problematic char slices about sent '
                                   f'from {begin}({bias[0] * base_factor}) to {end}({bias[1] * base_factor})')
                else:
                    logger.error(f'Sent: {sent.text}')
                    logger.error(f'Error char slices about sent '
                                 f'from {begin}({bias[0] * base_factor}) to {end}({bias[1] * base_factor})')
                break
        if all_fail:
            break
    if strs is None:
        logger.error(f'Sent: {sent.text}\n')
        logger.error(f'Error char slices about sent from {begin} to {end}')
        raise Exception('Illegal char slices')  # TODO: more specific Exception class

    return strs.start, strs.end


def get_token_idx(token: SpacyToken) -> int:
    return token.i - token.sent.start


def token_not_start(token: SpacyToken) -> bool:
    return token.i - token.sent.start != 0


def token_not_end(token: SpacyToken) -> bool:
    return token.sent.end - token.i > 1


def idx_valid(sent: SpacySpan, idx: Union[int, Sequence[int]], is_char=False) -> bool:
    token_valid: Callable[[int], bool] = lambda x: sent.start <= x < sent.end
    char_valid: Callable[[int], bool] = lambda x: sent.start_char <= x < sent.end_char
    valid = char_valid if is_char else token_valid
    if isinstance(idx, collections.abc.Sequence):
        for i in idx:
            if not valid(i):
                return False
        return True
    return valid(idx)
