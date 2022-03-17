from ..config import (
    ACTOR_DATA_DIR,
    ACTOR_LABEL,
    ACTOR_MODEL_NAME_OR_PATH,
    ACTOR_MODEL_TYPE,
    ACTOR_OUTPUT_DIR,
)
from .base import InferBase


class InferActor(InferBase):
    def __init__(self):
        super().__init__(
            ACTOR_DATA_DIR,
            ACTOR_MODEL_TYPE,
            ACTOR_MODEL_NAME_OR_PATH,
            ACTOR_OUTPUT_DIR,
            ACTOR_LABEL,
        )
