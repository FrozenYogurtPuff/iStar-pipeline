from typing import Union

from src.utils.typing import (
    BertEntityLabel,
    BertEntityLabelBio,
    FixEntityLabel,
)


def is_entity_type_ok(
    fix: FixEntityLabel, spa: Union[BertEntityLabelBio, BertEntityLabel]
) -> bool:
    if fix not in ["Actor", "Both", "Resource"]:
        raise Exception("Illegal FixEntityLabel " + fix)

    if fix == "Both":
        if spa == "O":
            return False
        return True
    return spa.endswith(fix)
