from typing import Callable, List, Literal, NamedTuple, Sequence, Tuple, Union

import spacy.tokens
from typing_extensions import TypeAlias, TypeGuard  # Before Python 3.10

# BERT
BertEntityLabelBio: TypeAlias = Literal[
    "O", "B-Actor", "I-Actor", "B-Resource", "I-Resource"
]
BertIntentionLabelBio: TypeAlias = Literal[
    "O", "B-Core", "I-Core", "B-Cond", "I-Cond", "B-Aux", "I-Aux"
]
BertUnionLabelBio: TypeAlias = Union[BertEntityLabelBio, BertIntentionLabelBio]

BertEntityLabel: TypeAlias = Literal["O", "Actor", "Resource"]
BertIntentionLabel: TypeAlias = Literal["O", "Core", "Cond", "Aux", "Quality"]
BertUnionLabel: TypeAlias = Union[BertEntityLabel, BertIntentionLabel]

Token: TypeAlias = str
# [[0.1, 0.8],
#  [0.9, 0.2]]
BertMatrix: TypeAlias = List[List[int]]

# Alignment
# [1]
# [1, 3]
Alignment: TypeAlias = List[int]

# Spacy
SpacyDoc: TypeAlias = spacy.tokens.Doc
SpacySpan: TypeAlias = spacy.tokens.Span
SpacyToken: TypeAlias = spacy.tokens.Token
HybridToken: TypeAlias = Union[SpacyToken, SpacySpan]

# JSONL Dataset
# [0, 3, "Actor"]
DatasetEntityLabel: TypeAlias = Tuple[int, int, BertEntityLabel]
DatasetIntentionLabel: TypeAlias = Tuple[int, int, BertIntentionLabel]
DatasetUnionLabel: TypeAlias = Tuple[int, int, BertUnionLabel]

# Rule
# Entity
FixEntityLabel: TypeAlias = Literal["Actor", "Resource", "Both"]
# ( [Student, Parents], "Actor" )
# ( [Student, tickets], ("Actor", "Resource") )
EntityRuleReturn: TypeAlias = Sequence[Tuple[HybridToken, FixEntityLabel]]
EntityRulePlugins: TypeAlias = Sequence[
    Callable[[SpacySpan], EntityRuleReturn]
]
# (Student, [1], [1, 2], "Actor")
EntityFix: TypeAlias = Tuple[HybridToken, Alignment, Alignment, FixEntityLabel]


# Intention
# TODO: really?
FixIntentionLabel: TypeAlias = Literal["Core", "Cond", "Aux", "Quality"]


# Slices for Aux
# [(1, 3), (2, 11)]  [], not [)
class SeqSlicesTuple(NamedTuple):
    start: int
    end: int
    type_: FixIntentionLabel


IntentionRuleAuxReturn: TypeAlias = List[HybridToken]
IntentionRuleAuxPlugins: TypeAlias = Sequence[
    Callable[[SpacySpan], IntentionRuleAuxReturn]
]
IntentionFix: TypeAlias = Tuple[
    HybridToken, Alignment, Alignment, FixIntentionLabel
]


# TypeGuard
def is_bert_entity_label_bio(val: str) -> TypeGuard[BertEntityLabelBio]:
    if val in ["O", "B-Actor", "I-Actor", "B-Resource", "I-Resource"]:
        return True
    return False


def is_bert_entity_label_bio_list(
    val: Sequence[str],
) -> TypeGuard[List[BertEntityLabelBio]]:
    return all(is_bert_entity_label_bio(v) for v in val)


def is_bert_intention_label_bio(val: str) -> TypeGuard[BertIntentionLabelBio]:
    if val in ["O", "B-Core", "I-Core", "B-Cond", "I-Cond", "B-Aux", "I-Aux"]:
        return True
    return False


def is_bert_intention_label_bio_list(
    val: Sequence[str],
) -> TypeGuard[List[BertIntentionLabelBio]]:
    return all(is_bert_intention_label_bio(v) for v in val)


def is_bert_union_label_bio(val: str) -> TypeGuard[BertUnionLabelBio]:
    return is_bert_entity_label_bio(val) or is_bert_intention_label_bio(val)


def is_bert_union_label_bio_list(
    val: Sequence[str],
) -> TypeGuard[List[BertUnionLabelBio]]:
    return is_bert_entity_label_bio_list(
        val
    ) or is_bert_intention_label_bio_list(val)


def is_bert_entity_label(val: str) -> TypeGuard[BertEntityLabel]:
    if val in ["O", "Actor", "Resource"]:
        return True
    return False


def is_bert_entity_label_list(
    val: Sequence[str],
) -> TypeGuard[List[BertEntityLabel]]:
    return all(is_bert_entity_label(v) for v in val)


def is_bert_intention_label(val: str) -> TypeGuard[BertIntentionLabel]:
    if val in ["O", "Core", "Cond", "Aux", "Quality"]:
        return True
    return False


def is_bert_intention_label_list(
    val: Sequence[str],
) -> TypeGuard[List[BertIntentionLabel]]:
    return all(is_bert_intention_label(v) for v in val)


def is_bert_union_label(val: str) -> TypeGuard[BertUnionLabel]:
    return is_bert_entity_label(val) or is_bert_intention_label(val)


def is_bert_union_label_list(
    val: Sequence[str],
) -> TypeGuard[List[BertUnionLabel]]:
    return is_bert_entity_label_list(val) or is_bert_intention_label_list(val)


def is_fix_entity_label(val: str) -> TypeGuard[FixEntityLabel]:
    if val in ["Actor", "Resource", "Both"]:
        return True
    return False


def is_fix_entity_label_list(
    val: Sequence[str],
) -> TypeGuard[List[FixEntityLabel]]:
    return all(is_fix_entity_label(v) for v in val)


# TODO: typing.NamedTuple 重构
# TODO: 使用继承 Exception 重构 LBYL
