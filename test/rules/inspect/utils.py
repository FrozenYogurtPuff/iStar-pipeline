from spacy import Language

from src.utils.typing import Span, Token

nlp_dict: dict[str, Span] = dict()

dep_list = [
    "acl",
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "attr",
    "aux",
    "auxpass",
    "case",
    "cc",
    "ccomp",
    "compound",
    "conj",
    "csubj",
    "csubjpass",
    "dative",
    "dep",
    "det",
    "dobj",
    "expl",
    "intj",
    "mark",
    "meta",
    "neg",
    "nmod",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "nummod",
    "oprd",
    "parataxis",
    "pcomp",
    "pobj",
    "poss",
    "preconj",
    "predet",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "relcl",
    "xcomp",
]

tag_list = [
    "ADD",
    "AFX",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NFP",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "XX",
]

ner_list = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]


def cache_nlp(nlp: Language, s: str) -> Span:
    global nlp_dict
    try:
        res = nlp_dict[s]
    except KeyError:
        res = nlp(s)[:]
        nlp_dict[s] = res
    return res


def if_inside(sentence: Span, dep: str) -> bool:
    for token in sentence:
        if token.dep_ == dep:
            return True
    return False


def format_markdown(t: Token | Span):
    ret = ""
    if isinstance(t, Span):
        t = t.root
    for tok in t.sent:
        if tok == t:
            ret += f"**{tok}** "
        else:
            ret += f"{tok} "
    return ret
