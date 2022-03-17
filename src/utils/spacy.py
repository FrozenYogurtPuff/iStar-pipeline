import collections.abc
import logging
from collections.abc import Sequence

import spacy
import spacy.tokens
from spacy.tokens import Doc, Span, Token

logger = logging.getLogger(__name__)

global_nlp = None


class IllegalCharSliceException(Exception):
    pass


def get_spacy():
    global global_nlp

    if global_nlp is None:
        global_nlp = spacy.load("en_core_web_lg")
    return global_nlp


def char_idx_to_word(sent: Span | Doc, begin: int, end: int) -> Span:
    biases = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    base_factor = 0
    strs = sent.char_span(begin, end)
    while strs is None:
        all_fail = True
        base_factor += 1
        for bias in biases:
            b, e = begin + bias[0] * base_factor, end + bias[1] * base_factor
            if not idx_valid(sent, (b, e), is_char=True):
                continue
            all_fail = False
            strs = sent.char_span(b, e)
            if strs is not None:
                if base_factor <= 1:
                    logger.warning(f"Sent: {sent.text}\n")
                    logger.warning(
                        f"Problematic char slices about sent "
                        f"from {begin}({bias[0] * base_factor}) to {end}({bias[1] * base_factor})"
                    )
                else:
                    logger.error(f"Sent: {sent.text}")
                    logger.error(
                        f"Error char slices about sent "
                        f"from {begin}({bias[0] * base_factor}) to {end}({bias[1] * base_factor})"
                    )
                break
        if all_fail:
            break
    if strs is None:
        logger.error(f"Sent: {sent.text}\n")
        logger.error(f"Error char slices about sent from {begin} to {end}")
        raise IllegalCharSliceException("Illegal char slices")
    return strs


# sent, 0, 14 -> 0, 2
def char_idx_to_word_idx(
    sent: Span | Doc, begin: int, end: int
) -> tuple[int, int]:
    strs = char_idx_to_word(sent, begin, end)
    return strs.start, strs.end


# [a, b], not (a, b)!
def get_token_idx(token: Token | Span) -> list[int]:
    def calc(t: Token):
        return t.i - t.sent.start

    if isinstance(token, spacy.tokens.Span):
        return [calc(token[0]), calc(token[-1])]
    return [calc(token)]


def token_not_start(token: Token | Span) -> bool:
    t = token[0] if isinstance(token, spacy.tokens.Span) else token
    ret: bool = t.i - t.sent.start != 0
    return ret


def token_not_end(token: Token | Span) -> bool:
    t = token[-1] if isinstance(token, spacy.tokens.Span) else token
    ret: bool = t.sent.end - t.i > 1
    return ret


def idx_valid(
    sent: Span | Doc, idx: int | Sequence[int], is_char=False
) -> bool:
    def token_valid(x: int) -> bool:
        ret: bool = sent[:].start <= x < sent[:].end
        return ret

    def char_valid(x: int) -> bool:
        ret: bool = sent[:].start_char <= x < sent[:].end_char
        return ret

    valid = char_valid if is_char else token_valid
    if isinstance(idx, collections.abc.Sequence):
        for i in idx:
            if not valid(i):
                return False
        return True
    return valid(idx)


def include_elem(elem: Token | Span, sents: Token | Span | Doc):
    def token_include_elem(el):
        if isinstance(sents, spacy.tokens.Token):
            return el == sents
        return el in sents

    if isinstance(elem, spacy.tokens.Span):
        return token_include_elem(elem[0]) and token_include_elem(elem[-1])
    return token_include_elem(elem)


def match_noun_chunk(token: Token | Span, sents: Span) -> Span | None:
    for noun_chunk in sents.noun_chunks:
        if include_elem(token, noun_chunk):
            return noun_chunk
    return None


def get_bio_sent_from_char_spans(
    sent: str, spans: list[Sequence[int | str]] | None = None
) -> tuple[list[str], list[str]]:
    nlp = get_spacy()
    doc = nlp(sent)

    temp = list()
    if spans:
        for start, end, label in spans:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(label, str)
            sw, ew = char_idx_to_word_idx(doc, start, end)
            temp.append(Span(doc, sw, ew, label))
    doc.set_ents(temp)

    result_label: list[str] = list()
    result_token: list[str] = list()
    for token in doc:
        if token.ent_iob_ == "O":
            result_label.append(token.ent_iob_)
        else:
            result_label.append(f"{token.ent_iob_}-{token.ent_type_}")
        result_token.append(token.text)
    return result_token, result_label
