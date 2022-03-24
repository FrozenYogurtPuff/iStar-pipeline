import json
from typing import Any

import spacy
from tqdm import tqdm

CONLL_FILE = "pretrained_data/2022/actor/divided/dev.txt"
ALL_JSONL = "pretrained_data/2022/actor/divided/all.jsonl"
SPLIT_OUTPUT = "pretrained_data/2022/actor/divided/split_dev.jsonl"

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    fp = open(SPLIT_OUTPUT, "w")

    cache: dict[tuple[str, ...], Any] = dict()

    with open(ALL_JSONL, "r") as jsonl:
        for line in tqdm(jsonl):
            a = json.loads(line)
            text = a["text"]
            sent = nlp(text)
            tokens = tuple(t.text for t in sent)
            cache[tokens] = a

    with open(CONLL_FILE, "r") as conll:
        result: list[str] = list()
        for line in tqdm(conll):
            if line != "\n":
                result.append(line.split()[0])
            else:
                key = tuple(tok for tok in result)
                try:
                    print(json.dumps(cache[key]), file=fp)
                except KeyError:
                    print(f"Not found {' '.join(key)}")
                result.clear()
        key = tuple(tok for tok in result)
        try:
            print(json.dumps(cache[key]), file=fp)
        except KeyError:
            print(f"Not found {' '.join(key)}")

    fp.close()
