from typing import List, Literal, Tuple, Callable, Sequence, Union
import spacy.tokens

# BERT
BertEntityLabel = Literal['O', 'B-Actor', 'I-Actor', 'B-Resource', 'I-Resource']
BertIntentionLabel = Literal['O', 'B-Core', 'I-Core', 'B-Cond', 'I-Cond', 'B-Aux', 'I-Aux']

BertEntityLabelRaw = Literal['O', 'Actor', 'Resource']
BertIntentionLabelRaw = Literal['O', 'Core', 'Cond', 'Aux', 'Quality']
BertLabelRaw = Union[BertEntityLabelRaw, BertIntentionLabelRaw]
FixEntityLabel = Literal['Actor', 'Resource', 'Both']

Token = str
# [[0.1, 0.8],
#  [0.9, 0.2]]
BertMatrix = List[List[int]]

# Alignment
# [1]
# [1, 3]
Alignment = List[int]

# Spacy
SpacySpan = spacy.tokens.Span
SpacyToken = spacy.tokens.Token

# JSONL Dataset
# [0, 3, "Actor"]
DatasetLabel = List[Union[int, int, BertLabelRaw]]

# Rule
# ( [Student, Parents], "Actor" )
# ( [Student, tickets], ("Actor", "Resource") )
EntityRuleReturn = Sequence[Tuple[SpacyToken, FixEntityLabel]]
EntityRulePlugins = Sequence[Callable[[SpacySpan], EntityRuleReturn]]
# (Student, 1, [1, 2], "Actor")
EntityFix = Tuple[SpacyToken, int, Alignment, FixEntityLabel]
