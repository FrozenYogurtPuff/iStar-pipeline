import logging

from spacy.tokens import Span, Token

from src.utils.spacy_utils import token_not_first

logger = logging.getLogger(__name__)


def acl_without_to(s: Span) -> list[Span | Token]:

    pool: list[Span | Token] = list()
    for token in s:
        if (
            token.dep_ == "acl"
            # `!=using`
            and token.lower_ not in ["using", "requiring"]
            and token_not_first(token.head)
            and token.head.nbor(1).lower_ != "to"
        ):
            if token_not_first(token):
                pool.append(token.nbor(-1))

    return pool
