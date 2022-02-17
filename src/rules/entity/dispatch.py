import logging
from typing import List

from src.typing import (
    Alignment,
    BertEntityLabel,
    EntityFix,
    SpacySpan,
    EntityRulePlugins
)

from ..config import entity_plugins


def dispatch(s: SpacySpan, b: List[BertEntityLabel],
             s2b: List[Alignment], funcs: EntityRulePlugins = entity_plugins) -> List[EntityFix]:
    unchecked = list()
    result = list()

    for func in funcs:
        li, label = func(s)
        for ul in li:
            unchecked.append((ul, label))

    remove_duplicate = set()
    for item in unchecked:
        token, label = item
        remove_duplicate.add((token.i - token.sent.start, label))

    for item in remove_duplicate:
        fix = False
        idx, label = item
        map_idx = s2b[idx]
        for num in map_idx:
            if not b[num].endswith(label):
                fix = True

        if fix:
            logging.getLogger(__name__).debug(f"{s.text}\n"
                                              f"token: {s[idx].text}, idx: {idx}, map_idx:{map_idx}, label: {label}")
            result.append((idx, map_idx, label))

    return result
