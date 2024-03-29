# lower_.startswith
# system 522/539
# 〃 [Actor] 481/539
# interface 78/82
# 〃 [Actor] 43/82
# 〃 [Resource] 35/82
# user 367/388
# 〃 [Actor] 306/388
# module 13/13
# 〃 [Actor] 11/13
# server 19/19
# 〃 [Actor] 17/19
# client 19/22
# 〃 [Actor] 11/22
# 〃 [Resource] 11/22
# administrator 69/72
# 〃 [Actor] 65/72
# Hybrid: 1038/1135

import logging

from src.utils.typing import Span


# system, interface, etc.
def word_list(s: Span):
    resource: str = "Resource"
    result = list()

    for token in s:
        select = False
        if token.lower_.startswith(
            ("system", "module", "server", "administrator")
        ):
            select = True
        elif token.lower_.startswith(("interface", "client", "user")):
            select = True

        if select:
            cur = (token, *token.conjuncts)
            for c in cur:
                result.append((c, resource))

    logging.getLogger(__name__).debug(f"Length {len(result)}: {result}")
    return result
