from typing import Union


class FixEntityLabelException(Exception):
    pass


def is_entity_type_ok(fix: str, spa: Union[str, str]) -> bool:
    if fix not in ["Actor", "Both", "Resource"]:
        raise FixEntityLabelException("Illegal FixEntityLabel " + fix)

    if fix == "Both":
        if spa == "O":
            return False
        return True
    return spa.endswith(fix)
