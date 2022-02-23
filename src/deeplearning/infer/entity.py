from ..config import (ENTITY_DATA_DIR, ENTITY_LABEL, ENTITY_MODEL_NAME_OR_PATH,
                      ENTITY_MODEL_TYPE, ENTITY_OUTPUT_DIR)
from .base import InferBase


class InferEntity(InferBase):
    def __init__(self):
        super().__init__(
            ENTITY_DATA_DIR,
            ENTITY_MODEL_TYPE,
            ENTITY_MODEL_NAME_OR_PATH,
            ENTITY_OUTPUT_DIR,
            ENTITY_LABEL
        )


entity_model = InferEntity()
