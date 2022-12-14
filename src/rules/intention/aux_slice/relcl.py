import logging

from spacy.tokens import Span, Token

from src.utils.spacy_utils import token_not_first, token_not_last

logger = logging.getLogger(__name__)


def relcl(s: Span) -> list[Span | Token]:

    pool: list[Span | Token] = list()
    for token in s:
        # `double != to`
        if (
            token.dep_ == "relcl"
            and token_not_first(token)
            and token.nbor(-1).lower_ != "to"
            and token_not_last(token.head)
            and token.head.nbor(1).lower_ != "to"
        ):
            pool.append(token.head)

    return pool
