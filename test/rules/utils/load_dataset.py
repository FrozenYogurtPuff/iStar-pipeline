import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

from src import ROOT_DIR


def load_dataset(
    path: str,
) -> Generator[tuple[int, str, list[Any]], None, None]:
    with open(
        Path(ROOT_DIR).joinpath(path),
        "r",
    ) as file:
        for idx, line in enumerate(file):
            a = json.loads(line)
            sent = a["text"]
            assert isinstance(sent, str)

            anno = list()
            for label in a["labels"]:
                s, e, lab = label
                anno.append((s, e, lab))
            yield idx, sent, anno
