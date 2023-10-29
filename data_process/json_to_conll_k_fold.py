import json
import os
from pathlib import Path

import spacy
from spacy.tokens import Span

from src.utils.spacy_utils import char_idx_to_word_idx

BASE = "../pretrained_data/2022_Kfold/intention/10/"
K = 10

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    INPUT = Path(BASE).joinpath("all.jsonl")
    fp_list = list()

    for i in range(K):
        os.makedirs(Path(BASE).joinpath(str(i)), exist_ok=True)
        fp_list.append(
            (
                open(Path(BASE).joinpath(str(i)).joinpath("train.txt"), "w"),
                open(Path(BASE).joinpath(str(i)).joinpath("dev.txt"), "w"),
                open(
                    Path(BASE).joinpath(str(i)).joinpath("split_dev.jsonl"),
                    "w",
                ),
            )
        )

    with open(INPUT, "r") as file:
        counter = 0
        for idx, line in enumerate(file):
            a = json.loads(line)
            text = a["text"]
            sent = nlp(text)
            anno = list()
            # for item in a["labels"]:
            for item in a["entities"]:
                # s, e, lab = item
                s, e, lab = (
                    item["start_offset"],
                    item["end_offset"],
                    item["label"],
                )
                start, end = char_idx_to_word_idx(sent[:], s, e)
                anno.append(Span(sent, start, end, lab))
            sent.set_ents(anno)
            for tok in sent:
                if tok.ent_type_ == "":
                    rep = f"{tok.text} O"
                else:
                    rep = f"{tok.text} {tok.ent_iob_}-{tok.ent_type_}"
                # 打印给 counter 的 dev [1] 与所有非 counter 的 train [0]
                for i in range(K):
                    if i == counter:
                        print(rep, file=fp_list[i][1])
                    else:
                        print(rep, file=fp_list[i][0])

            # 给 counter 的 dev 与所有非 counter 的 train 打印空格
            # 给 counter 的 split_dev 打印原始 json
            for i in range(K):
                if i == counter:
                    print(file=fp_list[i][1])
                    print(json.dumps(a), file=fp_list[i][2])
                else:
                    print(file=fp_list[i][0])

            counter = (counter + 1) % K

    for i in range(K):
        for j in range(3):
            fp_list[i][j].close()
