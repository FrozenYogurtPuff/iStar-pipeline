import logging

from spacy.tokens import Span, Token

logger = logging.getLogger(__name__)


def find_children(
    token: Token | Span, dep=None, pos=None, tag=None, text=None
):
    children_list: list[Token | Span] = list()
    if not list(token.children):
        return children_list
    for t in token.children:
        flag = True
        if dep and t.dep_ != dep:
            flag = False
        if pos and t.pos != pos:
            flag = False
        if tag and t.tag != tag:
            flag = False
        if text and t.lower_ != text:
            flag = False
        if flag:
            children_list.append(t)
    return children_list


def find_object(verb: Token | Span) -> list[Token]:
    # nsubjpass
    # dobj, pobj
    # prep -> pobj
    # extra: noun.conjuncts

    if isinstance(verb, Span):
        verb = verb.root

    noun_list = list()
    for child in [
        *find_children(verb, dep="nsubjpass"),
        *find_children(verb, dep="dobj"),
        *find_children(verb, dep="pobj"),
    ]:
        for item in (child, *child.conjuncts):
            noun_list.append(item)
    if not noun_list:
        for prep in find_children(verb, dep="prep"):
            for pobj in find_children(prep, dep="pobj"):
                for item in (pobj, *pobj.conjuncts):
                    if item not in noun_list:
                        noun_list.append(item)
    return noun_list
