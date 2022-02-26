from typing import Callable, List, Literal, Sequence, Tuple, Union

import spacy.tokens

# BERT
BertEntityLabelBio = Literal['O', 'B-Actor', 'I-Actor', 'B-Resource', 'I-Resource']
BertIntentionLabelBio = Literal['O', 'B-Core', 'I-Core', 'B-Cond', 'I-Cond', 'B-Aux', 'I-Aux']
BertUnionLabelBio = Union[BertEntityLabelBio, BertIntentionLabelBio]

BertEntityLabel = Literal['O', 'Actor', 'Resource']
BertIntentionLabel = Literal['O', 'Core', 'Cond', 'Aux', 'Quality']
BertUnionLabel = Union[BertEntityLabel, BertIntentionLabel]

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
DatasetEntityLabel = Tuple[int, int, BertEntityLabel]
DatasetIntentionLabel = Tuple[int, int, BertIntentionLabel]
DatasetUnionLabel = Union[DatasetEntityLabel, DatasetIntentionLabel]

# Rule
# ( [Student, Parents], "Actor" )
# ( [Student, tickets], ("Actor", "Resource") )
EntityRuleReturn = Sequence[Tuple[HybridToken, FixEntityLabel]]
EntityRulePlugins = Sequence[Callable[[SpacySpan], EntityRuleReturn]]
# (Student, [1], [1, 2], "Actor")
EntityFix = Tuple[HybridToken, Alignment, Alignment, FixEntityLabel]
