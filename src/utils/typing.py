from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, Union

if TYPE_CHECKING:
    from src.deeplearning.entity.infer.result import BertResult

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

# (Student, [1], [1, 2], "Actor")
class EntityFix(NamedTuple):
    token: Any
    idxes: Alignment
    bert_idxes: Alignment
    label: str


# Rule
# Entity
# ( [Student, Parents], "Actor" )
# ( [Student, tickets], ("Actor", "Resource") )
# RuleReturn: TypeAlias = Sequence[tuple[int, int, str]]
RuleReturn: TypeAlias = Sequence[EntityFix]
RulePlugin: TypeAlias = Callable[
    [Span, Union["BertResult", None], Union[list[Alignment], None]], RuleReturn
]
RulePlugins: TypeAlias = Sequence[RulePlugin]

# Relation
RelationReturn: TypeAlias = int | None
RelationPlugin: TypeAlias = Callable[[Span, Span, Span], RelationReturn]

# Slices for Aux
# [(1, 3), (2, 11)]  [], not [)
class SeqSlicesTuple(NamedTuple):
    start: int
    end: int
    type_: str


IntentionAuxReturn: TypeAlias = list[Token | Span]
IntentionAuxPlugins: TypeAlias = Sequence[Callable[[Span], IntentionAuxReturn]]
