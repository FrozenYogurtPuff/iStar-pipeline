from typing import List, Literal, Tuple
import spacy.tokens

# BERT
BertEntityLabel = Literal['O', 'B-Actor', 'I-Actor', 'B-Resource', 'I-Resource']
BertIntentionLabel = Literal['O', 'B-Core', 'I-Core', 'B-Cond', 'I-Cond', 'B-Aux', 'I-Aux']

BertEntityLabelRaw = Literal['O', 'Actor', 'Resource']
BertIntentionLabelRaw = Literal['O', 'Core', 'Cond', 'Aux', 'Quality']

Token = str
BertToken = List[Token]
BertEntityLabelList = List[BertEntityLabel]
BertIntentionLabelList = List[BertIntentionLabel]
BertMatrix = List[List[int]]

# Alignment
Alignment = List[int]
AlignmentList = List[Alignment]

EntityFix = Tuple[Alignment, BertEntityLabelRaw]
EntityFixList = List[EntityFix]

# Spacy
SpacySpan = spacy.tokens.Span
