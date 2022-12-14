# Infer - Actor
from pathlib import Path

from src import ROOT_DIR

ACTOR_DATA_DIR = str(Path(ROOT_DIR) / "pretrained_data/2022_Kfold/actor/0/")
ACTOR_MODEL_TYPE = "bert"
ACTOR_MODEL_NAME_OR_PATH = "bert-base-cased"
ACTOR_OUTPUT_DIR = str(
    Path(ROOT_DIR) / "pretrained_model/2022_Kfold/actor/0/output/"
)
ACTOR_LABEL = str(
    Path(ROOT_DIR) / "pretrained_data/2022_Kfold/actor/labels.txt"
)

ACTOR_COMBINED_DATA_DIR = str(
    Path(ROOT_DIR) / "pretrained_data/2022/actor/combined/"
)
ACTOR_COMBINED_MODEL_TYPE = "bert"
ACTOR_COMBINED_MODEL_NAME_OR_PATH = "bert-base-cased"
ACTOR_COMBINED_OUTPUT_DIR = str(
    Path(ROOT_DIR) / "pretrained_model/2022/actor/combined/"
)
ACTOR_COMBINED_LABEL = str(
    Path(ROOT_DIR) / "pretrained_data/2022/actor/combined/labels.txt"
)

# Infer - Resource
RESOURCE_DATA_DIR = str(Path(ROOT_DIR) / "pretrained_data/2022/resource/")
RESOURCE_MODEL_TYPE = "bert"
RESOURCE_MODEL_NAME_OR_PATH = "bert-base-cased"
RESOURCE_OUTPUT_DIR = str(Path(ROOT_DIR) / "pretrained_model/2022/resource/")
RESOURCE_LABEL = str(
    Path(ROOT_DIR) / "pretrained_data/2022/resource/labels.txt"
)

# Infer - Intention
INTENTION_DATA_DIR = str(Path(ROOT_DIR) / "pretrained_data/2022/task/verb/")
INTENTION_MODEL_TYPE = "bert"
INTENTION_MODEL_NAME_OR_PATH = "bert-base-cased"
INTENTION_OUTPUT_DIR = str(Path(ROOT_DIR) / "pretrained_model/2022/task/verb/")
INTENTION_LABEL = str(
    Path(ROOT_DIR) / "pretrained_data/2022/task/verb/labels.txt"
)
