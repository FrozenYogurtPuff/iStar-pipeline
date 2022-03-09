import logging
from typing import List

from src.utils.typing import HybridToken, SpacySpan

logger = logging.getLogger(__name__)


def agent(s: SpacySpan) -> List[HybridToken]:

    pool: List[HybridToken] = list()
    for token in s:
        if (
            token.dep_
            == "agent"
            # and token_not_start(token)
            # and token.nbor(-1).lower_ != "to"
            # and token_not_end(token.head)
            # and token.head.nbor(1).lower_ != "to"
        ):
            pool.append(token.head)

    return pool
