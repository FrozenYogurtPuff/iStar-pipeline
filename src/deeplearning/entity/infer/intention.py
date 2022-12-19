from ..config import (
    INTENTION_DATA_DIR,
    INTENTION_LABEL,
    INTENTION_MODEL_NAME_OR_PATH,
    INTENTION_MODEL_TYPE,
    INTENTION_OUTPUT_DIR,
)
from .base import InferBase


class InferIntention(InferBase):
    def __init__(self, data=None, model=None, label=None):
        super().__init__(
            INTENTION_DATA_DIR if not data else data,
            INTENTION_MODEL_TYPE,
            INTENTION_MODEL_NAME_OR_PATH,
            INTENTION_OUTPUT_DIR if not model else model,
            INTENTION_LABEL if not label else label,
        )
