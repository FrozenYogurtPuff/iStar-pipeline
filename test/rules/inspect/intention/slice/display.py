import logging
from pathlib import Path
from typing import Any

from spacy import displacy, tokens
from spacy.tokens import Doc, Span

from src.utils.spacy_utils import char_idx_to_word_idx, get_spacy

logger = logging.getLogger(__name__)


def if_inside(sentence: Span) -> bool:
    # TODO:
    return True


def display_dep(doc: Doc, idx: int) -> None:
    svg = displacy.render(doc, style="dep")
    output_path = Path(f"visualize/{idx}_dep.svg")
    output_path.open("w", encoding="utf-8").write(svg)


def display_ent(doc: Doc, annotates: Any, idx: int) -> None:
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
        start_char, end_char, type_ = annotate
        start, end = char_idx_to_word_idx(doc[:], start_char, end_char)
        true_ents.append(tokens.Span(doc, start, end, type_))
    doc.ents = true_ents
    svg = displacy.render(doc, style="ent", options=options)
    output_path = Path(f"visualize/{idx}_ent.html")
    output_path.open("w", encoding="utf-8").write(svg)


if __name__ == "__main__":
    # data = list(load_dataset("pretrained_data/task_core_aux_cond/all.jsonl"))
    # data = list(load_dataset("pretrained_data/2022/task/all/all.jsonl"))
    # data = list(load_dataset("pretrained_data/2022/actor/divided/all.jsonl"))
    nlp = get_spacy()
    display_dep(
        nlp(
            "This feature will allow the user or administrator to view details."
        ),
        114514,
    )
    #
    # for i, sent, anno in tqdm(data):
    #     sent_processed = nlp(sent)
    #     if if_inside(sent_processed[:]):
    #         # display_dep(sent_processed, i)
    #         display_ent(sent_processed, anno, i)
    #         # print(i)

# TODO: advcl 与 Cond
# TODO: nsubj时查找verb的conj进行分句
