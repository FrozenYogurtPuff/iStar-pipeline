from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from spacy import tokens, displacy
from spacy.tokens import Doc

from src.deeplearning.entity.infer.utils import get_series_bio
from src.deeplearning.entity.infer.wrapper import ActorWrapper, \
    IntentionWrapper
from src.utils.spacy_utils import char_idx_to_word_idx, get_spacy
from test.rules.utils.load_dataset import load_dataset


# 生成句中实体的 svg
def display(doc: Doc, annotates: Any, idx: int, options: dict, desc: str | None = None) -> None:
    true_ents = list()
    for annotate in annotates:
        start_char, end_char, type_ = annotate
        start, end = char_idx_to_word_idx(doc[:], start_char, end_char)
        true_ents.append(tokens.Span(doc, start, end, type_))
    doc.ents = true_ents
    svg = displacy.render(doc, style="ent", options=options)
    if desc:
        output_path = Path(f"visualize/{idx}_{desc}.html")
    else:
        output_path = Path(f"visualize/{idx}.html")
    output_path.open("w", encoding="utf-8").write(svg)


def display_actor(doc: Doc, annotates: Any, idx: int, desc: str | None = None):
    colors = {
        "Agent": "#85C1E9",
        "Role": "#ff6961",
    }
    options = {"ents": ["Agent", "Role"], "colors": colors}
    display(doc, annotates, idx, options, desc)


def display_intention_core(doc: Doc, annotates: Any, idx: int, desc: str | None = None):
    colors = {
        "Core": "#85C1E9",
    }
    options = {"ents": ["Core"], "colors": colors}
    display(doc, annotates, idx, options, desc)


# 查找 trues 与 preds 的差异，返回是否有区别的 bool 值
def is_different(true_entities, pred_entities, only_check: Sequence[str] | None = None) -> bool:
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        # Select all
        if (not only_check) or (e[0] in only_check):
            d1[e[0]].add((e[1], e[2]))
    for e in pred_entities:
        if (not only_check) or (e[0] in only_check):
            d2[e[0]].add((e[1], e[2]))

    for type_name, type_true_entities in d1.items():
        type_pred_entities = d2[type_name]
        nb_diff = type_true_entities - (
            type_true_entities & type_pred_entities
        )
        if nb_diff:
            return True
        nb_diff = type_pred_entities - (
            type_true_entities & type_pred_entities
        )
        if nb_diff:
            return True
    return False


# 可视化错误项目的 pred 和 true，便于查找 include 与 exclude 规则
def visualize_actor_pure_bert():
    data = list(
        load_dataset("pretrained_data/2022/actor/divided/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = ActorWrapper()
    nlp = get_spacy()

    # Make sure only one single sent
    for i, (sent, label) in enumerate(zip(sents, labels)):
        result = wrapper.process(sent, label)
        pred_entitie, true_entitie = get_series_bio(result)

        if is_different(true_entitie, pred_entitie):
            sent_processed = nlp(sent)
            display_actor(sent_processed, label, i, "true")

            # pred show
            pred_label = list()
            for pred in pred_entitie:
                span = sent_processed[pred[1]:pred[2] + 1]
                pred_label.append((span.start_char, span.end_char, pred[0]))
            display_actor(sent_processed, pred_label, i, "pred")


def visualize_intention_pure_bert():
    data = list(
        load_dataset("pretrained_data/2022/task/verb/split_dev.jsonl")
    )
    sents = [d[1] for d in data]
    labels = [d[2] for d in data]

    wrapper = IntentionWrapper()
    nlp = get_spacy()

    # Make sure only one single sent
    for i, (sent, label) in enumerate(zip(sents, labels)):
        result = wrapper.process(sent, label)
        pred_entitie, true_entitie = get_series_bio(result)

        if is_different(true_entitie, pred_entitie, ["Core"]):
            sent_processed = nlp(sent)
            display_intention_core(sent_processed, label, i, "true")

            # pred show
            pred_label = list()
            for pred in pred_entitie:
                span = sent_processed[pred[1]:pred[2] + 1]
                pred_label.append((span.start_char, span.end_char, pred[0]))
            display_intention_core(sent_processed, pred_label, i, "pred")


if __name__ == '__main__':
    visualize_intention_pure_bert()
