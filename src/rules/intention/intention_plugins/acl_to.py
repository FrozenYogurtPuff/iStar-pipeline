# slightly approve
import logging

from spacy.tokens import Span


def acl_to(s: Span):
    core: str = "Core"
    result = list()

    for token in s:
        if token.dep_ == "acl":
            for child in token.children:
                if child.dep_ == "aux" and child.lower_ == "to":
                    result.append((token, core))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
