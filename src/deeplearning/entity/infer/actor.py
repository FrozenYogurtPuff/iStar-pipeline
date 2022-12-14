from ..config import (
    ACTOR_COMBINED_DATA_DIR,
    ACTOR_COMBINED_LABEL,
    ACTOR_COMBINED_MODEL_NAME_OR_PATH,
    ACTOR_COMBINED_MODEL_TYPE,
    ACTOR_COMBINED_OUTPUT_DIR,
    ACTOR_DATA_DIR,
    ACTOR_LABEL,
    ACTOR_MODEL_NAME_OR_PATH,
    ACTOR_MODEL_TYPE,
    ACTOR_OUTPUT_DIR,
)
from .base import InferBase


class InferActor(InferBase):
    def __init__(self, data=None, model=None, label=None):
        super().__init__(
            ACTOR_DATA_DIR if not data else data,
            ACTOR_MODEL_TYPE,
            ACTOR_MODEL_NAME_OR_PATH,
            ACTOR_OUTPUT_DIR if not model else model,
            ACTOR_LABEL if not label else label,
        )


class InferCombinedActor(InferBase):
    def __init__(self):
        super().__init__(
            ACTOR_COMBINED_DATA_DIR,
            ACTOR_COMBINED_MODEL_TYPE,
            ACTOR_COMBINED_MODEL_NAME_OR_PATH,
            ACTOR_COMBINED_OUTPUT_DIR,
            ACTOR_COMBINED_LABEL,
        )
