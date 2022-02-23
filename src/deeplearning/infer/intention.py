from ..config import (INTENTION_DATA_DIR, INTENTION_LABEL,
                      INTENTION_MODEL_NAME_OR_PATH, INTENTION_MODEL_TYPE,
                      INTENTION_OUTPUT_DIR)
from .base import InferBase


class InferIntention(InferBase):
    def __init__(self):
        super().__init__(
            INTENTION_DATA_DIR,
            INTENTION_MODEL_TYPE,
            INTENTION_MODEL_NAME_OR_PATH,
            INTENTION_OUTPUT_DIR,
            INTENTION_LABEL
        )


intention_model = InferIntention()
