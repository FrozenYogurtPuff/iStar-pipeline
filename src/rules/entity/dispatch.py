import logging
from typing import List

from src.typing import (
    Alignment,
    BertEntityLabel,
    EntityFix,
    SpacySpan,
    EntityRulePlugins
)
from src.rules.utils.seq import is_entity_type_ok
from src.rules.utils.spacy import get_token_idx

from ..config import entity_plugins


def dispatch(s: SpacySpan, b: List[BertEntityLabel],
             s2b: List[Alignment], add_all: bool = False,
             funcs: EntityRulePlugins = entity_plugins) -> List[EntityFix]:
    unchecked = list()
    both = list()
    result = list()

    for func in funcs:
        packs = func(s)  # TODO: label can be resolved as tuple

        for token, label in set(packs):
            if label == 'Both':
                both.append(token)
            else:
                unchecked.append((token, label))

    # Add both sign 'Both'
    for token in both:
        for token_hat, _ in unchecked:
            if token == token_hat:
                break
        unchecked.append((token, 'Both'))

    for token, label in unchecked:
        idx = get_token_idx(token)
        if add_all:
            logging.getLogger(__name__).debug(f"{s.text}\n"
                                              f"token: {token.text}, idx: {idx}, label: {label}")
            result.append((token, idx, [], label))
        else:
            select = False
            map_idx = s2b[idx]
            for num in map_idx:
                if not is_entity_type_ok(label, b[num]):
                    select = True
            if select:
                logging.getLogger(__name__).debug(f"{s.text}\n"
                                                  f"token: {token.text}, idx: {idx},"
                                                  f"map_idx:{map_idx}, label: {label}")
                result.append((token, idx, map_idx, label))

    return result
