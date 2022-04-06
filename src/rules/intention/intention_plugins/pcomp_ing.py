# slightly disapprove
import logging

from spacy.tokens import Span

from src.utils.typing import RuleReturn


# Mashbot will provide an interface for authenticating a user account to an external service account .
def pcomp_ing(s: Span) -> RuleReturn:
    core: str = "Core"
    result = list()

    for token in s:
        if token.dep_ == "pcomp":
            if token.lower_.endswith("ing"):
                result.append((token, core))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
