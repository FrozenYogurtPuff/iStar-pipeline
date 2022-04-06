from collections.abc import Callable, Sequence
from typing import NamedTuple, TypeAlias

from spacy.tokens import Span, Token

# [[0.1, 0.8],
#  [0.9, 0.2]]
BertMatrix: TypeAlias = list[list[int]]

# Alignment
# [1]
# [1, 3]
Alignment: TypeAlias = list[int]

# JSONL Dataset
# [0, 3, "Actor"]
DatasetEntityLabel: TypeAlias = tuple[int, int, str]
DatasetIntentionLabel: TypeAlias = tuple[int, int, str]
DatasetUnionLabel: TypeAlias = tuple[int, int, str]

# Rule
# Entity
# ( [Student, Parents], "Actor" )
# ( [Student, tickets], ("Actor", "Resource") )
RuleReturn: TypeAlias = Sequence[tuple[Token | Span, str]]
RulePlugins: TypeAlias = Sequence[Callable[[Span], RuleReturn]]


# (Student, [1], [1, 2], "Actor")
class EntityFix(NamedTuple):
    token: Token | Span
    idxes: Alignment
    bert_idxes: Alignment
    label: str


# Slices for Aux
# [(1, 3), (2, 11)]  [], not [)
class SeqSlicesTuple(NamedTuple):
    start: int
    end: int
    type_: str


IntentionAuxReturn: TypeAlias = list[Token | Span]
IntentionAuxPlugins: TypeAlias = Sequence[Callable[[Span], IntentionAuxReturn]]
