import logging

from src.utils.typing import Span, Token

logger = logging.getLogger(__name__)


def agent(s: Span) -> list[Span | Token]:

    pool: list[Span | Token] = list()
    for token in s:
        if (
            token.dep_
            == "agent"
            # and token_not_first(token)
            # and token.nbor(-1).lower_ != "to"
            # and token_not_last(token.head)
            # and token.head.nbor(1).lower_ != "to"
        ):
            pool.append(token.head)

    return pool
