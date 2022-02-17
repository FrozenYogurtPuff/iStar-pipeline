import logging
from typing import Tuple

from src.typing import SpacySpan, SpacyToken

logger = logging.getLogger(__name__)


# sent, 0, 14 -> 0, 2
def char_idx_to_word_idx(sent: SpacySpan, begin: int, end: int) -> Tuple[int, int]:
    biases = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    strs = sent.char_span(begin, end)
    if strs is None:
        for b in biases:
            strs = sent.char_span(begin + b[0], end + b[1])
            if strs:
                logger.warning(f'Sent: {sent.text}\n'
                               f'Problematic char slices about sent from {begin}({b[0]}) to {end}({b[1]})')
                return strs.start, strs.end
        if strs is None:
            logger.error(f'Sent: {sent.text}\n'
                         f'Error char slices about sent from {begin} to {end}')
            raise Exception('Illegal char slices')

    return strs.start, strs.end


def get_token_idx(token: SpacyToken) -> int:
    return token.i - token.sent.start
