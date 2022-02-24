from typing import Callable, List, Literal, Sequence, Tuple, Union

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
SpacyDoc = spacy.tokens.Doc
SpacySpan = spacy.tokens.Span
SpacyToken = spacy.tokens.Token
HybridToken = Union[SpacyToken, SpacySpan]

# JSONL Dataset
# [0, 3, "Actor"]
DatasetEntityLabel = Tuple[int, int, BertEntityLabelRaw]

# Rule
# ( [Student, Parents], "Actor" )
# ( [Student, tickets], ("Actor", "Resource") )
EntityRuleReturn = Sequence[Tuple[HybridToken, FixEntityLabel]]
EntityRulePlugins = Sequence[Callable[[SpacySpan], EntityRuleReturn]]
# (Student, [1], [1, 2], "Actor")
EntityFix = Tuple[HybridToken, Alignment, Alignment, FixEntityLabel]
