import json
from pathlib import Path
from typing import Any, Generator, List, Tuple

from src import ROOT_DIR
from src.utils.typing import is_bert_union_label


def load_dataset(
    path: str,
) -> Generator[Tuple[int, str, List[Any]], None, None]:  # TODO: dirty fix
    with open(
        Path(ROOT_DIR).joinpath(path),
        "r",
    ) as j:
        for idx, line in enumerate(j):
            a = json.loads(line)
            sent = a["text"]
            assert isinstance(sent, str)
            anno = list()
            for label in a["labels"]:
                s, e, lab = label
                assert isinstance(s, int)
                assert isinstance(e, int)
                assert is_bert_union_label(lab)
                anno.append((s, e, lab))
            yield idx, sent, anno
