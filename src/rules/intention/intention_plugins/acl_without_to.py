# negative
import logging

from spacy.tokens import Span

from src.utils.typing import RuleReturn


def acl_without_to(s: Span) -> RuleReturn:
    aux: str = "Aux"
    result = list()

    for token in s:
        if token.dep_ == "acl":
            flag = True
            for child in token.children:
                if child.dep_ == "aux" and child.lower_ == "to":
                    flag = False

            if flag:
                result.append((token, aux))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
