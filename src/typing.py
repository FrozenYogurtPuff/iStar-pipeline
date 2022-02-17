from typing import List, Literal, Tuple, Callable, Sequence
import spacy.tokens

# BERT
BertEntityLabel = Literal['O', 'B-Actor', 'I-Actor', 'B-Resource', 'I-Resource']
BertIntentionLabel = Literal['O', 'B-Core', 'I-Core', 'B-Cond', 'I-Cond', 'B-Aux', 'I-Aux']

BertEntityLabelRaw = Literal['O', 'Actor', 'Resource']
BertIntentionLabelRaw = Literal['O', 'Core', 'Cond', 'Aux', 'Quality']

Token = str
BertMatrix = List[List[int]]

# Alignment
Alignment = List[int]

# Spacy
SpacySpan = spacy.tokens.Span
SpacyToken = spacy.tokens.Token

# Rule
EntityRuleReturn = Tuple[List[SpacyToken], BertEntityLabelRaw]
EntityRulePlugins = Sequence[Callable[[SpacySpan], EntityRuleReturn]]
EntityFix = Tuple[int, Alignment, BertEntityLabelRaw]
