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
            if "labels" in a:
                for label in a["labels"]:
                    s, e, lab = label
                    anno.append((s, e, lab))
            elif "label" in a:
                for label in a["label"]:
                    s, e, lab = label
                    anno.append((s, e, lab))
            elif "entities" in a:
                for item in a["entities"]:
                    s, e, lab = (
                        item["start_offset"],
                        item["end_offset"],
                        item["label"],
                    )
                    anno.append((s, e, lab))
            else:
                raise Exception("Illegal dataset format")
            yield idx, sent, anno
