import logging
from typing import Optional

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
    def get_nsubj(token: Token) -> Optional[Token]:
        children_list = find_children(token, dep="nsubj")
        if not children_list:
            return None
        return children_list[0]

    if isinstance(verb, Span):
        verb = verb.root
    verbs = [verb]

    verb_nsubj = get_nsubj(verb)
    for v in verb.conjuncts:
        v_nsubj = get_nsubj(v)
        if v_nsubj == verb_nsubj:
            verbs.append(v)
        elif verb_nsubj and not v_nsubj:
            verbs.append(v)

    # noun_list = list()
    # for child in [
    #     *find_children(verb, dep="nsubjpass"),
    #     *find_children(verb, dep="dobj"),
    #     *find_children(verb, dep="pobj"),
    # ]:
    #     for item in (child, *child.conjuncts):
    #         noun_list.append(item)
    # if not noun_list:
    #     for prep in find_children(verb, dep="prep"):
    #         for pobj in find_children(prep, dep="pobj"):
    #             for item in (pobj, *pobj.conjuncts):
    #                 if item not in noun_list:
    #                     noun_list.append(item)
    # return noun_list

    return_list = list()
    collector = list(verbs)
    while collector:
        item = collector.pop()
        return_list.append(item)
        if (
            item.dep_ in ["acomp"]  # , "ccomp", "xcomp"
            and item.has_head()
            and item.head not in return_list
            and item.head not in collector
        ):
            collector.append(item.head)
        if (
            item.pos_ == "PART"
            and item.has_head()
            and item.head not in return_list
            and item.head not in collector
        ):
            collector.append(item.head)

        for child in [
            *find_children(item, dep="nsubjpass"),
            *find_children(item, dep="auxpass"),
            *find_children(item, dep="dobj"),
            *find_children(item, dep="pobj"),
            *find_children(item, dep="advmod"),
            *find_children(item, dep="neg"),
            # *find_children(item, dep="xcomp"),
            # *find_children(item, dep="ccomp"),
            *find_children(item, dep="acomp"),
            # *find_children(item, dep="pcomp"),
            *find_children(item, dep="prep"),
        ]:
            for it in (child, *child.conjuncts):
                if it not in return_list and it not in collector:
                    collector.append(it)

    return return_list
