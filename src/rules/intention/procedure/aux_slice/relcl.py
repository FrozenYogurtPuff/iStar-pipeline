import logging
from typing import List

from src.utils.spacy import token_not_end, token_not_start
from src.utils.typing import HybridToken, SpacySpan

logger = logging.getLogger(__name__)


def relcl(s: SpacySpan) -> List[HybridToken]:

    pool: List[HybridToken] = list()
    for token in s:
        # `double != to`
        if (
            token.dep_ == "relcl"
            and token_not_start(token)
            and token.nbor(-1).lower_ != "to"
            and token_not_end(token.head)
            and token.head.nbor(1).lower_ != "to"
        ):
            pool.append(token.head)

    return pool
