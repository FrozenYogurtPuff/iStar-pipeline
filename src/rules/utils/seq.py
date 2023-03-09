import logging

from src.utils.typing import Alignment

logger = logging.getLogger(__name__)


class FixEntityLabelException(Exception):
    pass


# 判定一个函数是否「相同」
def is_entity_type_ok(fix: str, spa: str) -> bool:
    if fix not in [
        "Actor",
        "Both",
        "Role",
        "Resource",
        "Core",
        "Aux",
        "Cond",
        "Agent",
    ]:
        raise FixEntityLabelException("Illegal FixEntityLabel " + fix)

    if fix == "Both":
        if spa == "O":
            return False
        return True
    return spa.endswith(fix)


def get_s2b_idx(s2b: list[Alignment], idx: list[int]) -> list[int]:
    def find_last_idx(align: list[Alignment]) -> int:
        for item in align[::-1]:
            if item:
                return item[-1]
        return len(s2b)

    if len(idx) not in [1, 2]:
        raise Exception(f"Illegal idx list length: {len(idx)}")

    prev = s2b[idx[0]][0] - 1 if s2b[idx[0]] else None

    if prev is None:
        prev_idx = idx[0] - 1
        while prev_idx >= 0 and not s2b[prev_idx]:
            prev_idx -= 1
        prev = s2b[prev_idx][-1] if prev_idx >= 0 else -1

    next_idx = idx[-1] + 1
    while next_idx < len(s2b) and not s2b[next_idx]:
        next_idx += 1
    next_ = s2b[next_idx][0] if next_idx < len(s2b) else find_last_idx(s2b) + 1

    if prev + 1 > next_ - 1:
        return []
    else:
        return [prev + 1, next_ - 1]
