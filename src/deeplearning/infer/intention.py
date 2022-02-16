from .base import InferBase
from ..config import (
    INTENTION_LABEL,
    INTENTION_DATA_DIR,
    INTENTION_MODEL_TYPE,
    INTENTION_OUTPUT_DIR,
    INTENTION_MODEL_NAME_OR_PATH)


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
