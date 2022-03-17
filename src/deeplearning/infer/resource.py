from ..config import (
    RESOURCE_DATA_DIR,
    RESOURCE_LABEL,
    RESOURCE_MODEL_NAME_OR_PATH,
    RESOURCE_MODEL_TYPE,
    RESOURCE_OUTPUT_DIR,
)
from .base import InferBase


class InferResource(InferBase):
    def __init__(self):
        super().__init__(
            RESOURCE_DATA_DIR,
            RESOURCE_MODEL_TYPE,
            RESOURCE_MODEL_NAME_OR_PATH,
            RESOURCE_OUTPUT_DIR,
            RESOURCE_LABEL,
        )
