from ..config import (
    ENTITY_DATA_DIR,
    ENTITY_LABEL,
    ENTITY_MODEL_NAME_OR_PATH,
    ENTITY_MODEL_TYPE,
    ENTITY_OUTPUT_DIR,
)
from .base import InferBase

global_entity_model = None


class InferEntity(InferBase):
    def __init__(self):
        super().__init__(
            ENTITY_DATA_DIR,
            ENTITY_MODEL_TYPE,
            ENTITY_MODEL_NAME_OR_PATH,
            ENTITY_OUTPUT_DIR,
            ENTITY_LABEL,
        )


def get_entity_model():
    global global_entity_model

    if global_entity_model is None:
        global_entity_model = InferEntity()
    return global_entity_model
