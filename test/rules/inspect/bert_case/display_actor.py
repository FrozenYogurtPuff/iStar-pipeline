import logging
import pickle
from pathlib import Path
from test.rules.utils.load_dataset import load_dataset
from typing import Any

from spacy import displacy, tokens
from spacy.tokens import Doc
from tqdm import tqdm

from src import ROOT_DIR
from src.deeplearning.entity.infer.utils import get_series_bio
from src.utils.spacy_utils import get_spacy

logger = logging.getLogger(__name__)


def display_ent(
    doc: Doc, annotates: Any, fold: int, idx: int, true: bool
) -> None:
    # colors = {
    #     "Core": "#85C1E9",
    #     "Aux": "#ff6961",
    #     "Cond": "#ff6961",
    #     "Quality": "#85C1E9",
    # }
    # options = {"ents": ["Core", "Aux", "Cond", "Quality"], "colors": colors}
    colors = {
        "Agent": "#85C1E9",
        "Role": "#ff6961",
    }
    options = {"ents": ["Agent", "Role"], "colors": colors}

    true_ents = list()
    for annotate in annotates:
        type_, start, end = annotate
        # start, end = char_idx_to_word_idx(doc[:], start_char, end_char)
        true_ents.append(tokens.Span(doc, start, end + 1, type_))
    doc.ents = true_ents
    svg = displacy.render(doc, style="ent", options=options)
    if true:
        output_path = Path(f"visualize/{idx}_{fold}_gt_ent.html")
    else:
        output_path = Path(f"visualize/{idx}_{fold}_pred_ent.html")
    output_path.open("w", encoding="utf-8").write(svg)


if __name__ == "__main__":
    nlp = get_spacy()
    for i in tqdm(range(10)):
        data = list(
            load_dataset(
                f"pretrained_data/2022_Kfold/actor/10/{i}/split_dev.jsonl"
            )
        )
        sents = [d[1] for d in data]
        labels = [d[2] for d in data]

        data2 = str(
            Path(ROOT_DIR) / f"pretrained_data/2022_Kfold/actor/10/{i}/"
        )
        model = str(
            Path(ROOT_DIR) / f"pretrained_model/2022_Kfold/actor/10/{i}/"
        )
        label = str(
            Path(ROOT_DIR) / "pretrained_data/2022_Kfold/actor/10/labels.txt"
        )

        with open(f"cache/ae_bert_{i}.bin", "rb") as file:
            results = pickle.load(file)

        for idx, item in enumerate(results):
            pred_entities, true_entities = get_series_bio([item])
            if pred_entities != true_entities:
                sent_processed = nlp(sents[idx])
                display_ent(sent_processed, true_entities, i, idx, True)
                display_ent(sent_processed, pred_entities, i, idx, False)
