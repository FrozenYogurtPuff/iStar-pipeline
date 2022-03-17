import json
import random

import spacy
from spacy.tokens import Span

from src.utils.spacy import char_idx_to_word_idx

INPUT = "../pretrained_data/2022/resource/all.jsonl"
OUTPUT1 = "../pretrained_data/2022/resource/train.txt"
OUTPUT2 = "../pretrained_data/2022/resource/dev.txt"
PROPORTION = 80

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    fp1 = open(OUTPUT1, "w")
    fp2 = open(OUTPUT2, "w")

    with open(INPUT, "r") as file:
        for idx, line in enumerate(file):
            dice = random.randint(1, 100)
            if dice > PROPORTION:
                fp = fp2
            else:
                fp = fp1

            a = json.loads(line)
            text = a["text"]
            sent = nlp(text)
            anno = list()
            for label in a["labels"]:
                s, e, lab = label
                start, end = char_idx_to_word_idx(sent[:], s, e)
                anno.append(Span(sent, start, end, lab))
            sent.set_ents(anno)
            for tok in sent:
                if tok.ent_type_ == "":
                    rep = f"{tok.text} O"
                else:
                    rep = f"{tok.text} {tok.ent_iob_}-{tok.ent_type_}"
                print(rep, file=fp)
            print(file=fp)
