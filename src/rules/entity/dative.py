import logging

from src.typing import AlignmentList, BertEntityLabelRaw, BertEntityLabelList, EntityFixList, SpacySpan


# Show things to [Anna].
# to (ADP, dative) -> Anna (pobj)
def dative(s: SpacySpan, b: BertEntityLabelList, s2b: AlignmentList) -> EntityFixList:
    unchecked = list()
    result = list()
    actor: BertEntityLabelRaw = 'Actor'

    for token in s:
        if token.pos_ == 'ADP' and token.dep_ == 'dative':
            key = list(token.children)
            for k in key:
                if k.dep_ == 'pobj':
                    unchecked.append((k, *k.conjuncts))

    for li in unchecked:
        for item in li:
            idx = s2b[item.i - item.sent.start]
            fix = False
            for num in idx:
                if not b[num].endswith('Actor'):
                    fix = True

            if fix:
                logging.getLogger(__name__).debug(f"{s.text}\n{[i.text for i in li]}\t[{item.text}]")
                result.append((idx, actor))

        return result
