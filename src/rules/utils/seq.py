from typing import Union

from src.utils.typing import FixEntityLabel, BertEntityLabel, BertEntityLabelRaw


def is_entity_type_ok(fix: FixEntityLabel, spacy: Union[BertEntityLabel, BertEntityLabelRaw]) -> bool:
    if fix not in ['Actor', 'Both', 'Resource']:
        raise Exception('Illegal FixEntityLabel ' + fix)

    if fix == 'Both':
        if spacy == 'O':
            return False
        return True
    return spacy.endswith(fix)
