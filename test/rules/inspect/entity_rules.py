# # Date: 2023/03/16 Re-check useful rules before case study
# import logging
# import pickle
#
# # from test.rules.actor.test_entity_rules_precision import check_result_precision
# from test.rules.utils.load_dataset import load_dataset
#
# import spacy
#
# from src.deeplearning.entity.infer.result import BertResult
# from src.deeplearning.entity.infer.utils import get_series_bio
# from src.rules.dispatch import dispatch
# from src.utils.spacy_utils import get_token_idx, token_not_last
# from src.utils.typing import EntityFix, RulePlugin, Span, Token
#
# logger = logging.getLogger(__name__)
#
# both: str = "Both"
# agent: str = "Agent"
# role: str = "Role"
# core: str = "Core"
#
#
# def not_in_head(t: Token, default: list[str] = None) -> bool:
#     if not default:
#         default = ["acl", "relcl", "ccomp"]
#     while t.dep_ != "ROOT":
#         t = t.head
#         if t.dep_ in default:
#             return False
#     return True
#
#
# def ae_exclude_pron(s: Span):
#     result = list()
#     o: str = "O"
#     for tok in s:
#         if tok.pos_ == "PRON":
#             result.append((tok, o))
#
#     return result
#
#
# def dative_propn(s: Span):
#     result = list()
#
#     for token in s:
#         if token.pos_ in ["PROPN", "NOUN"] and token.dep_ == "dative":
#             cur = (token, *token.conjuncts)
#             for c in cur:
#                 result.append((c, both))
#
#     return result
#
#
# def relcl_who(s: Span):
#     result = list()
#
#     for token in s:
#         if token.dep_ == "relcl":
#             key = token.head
#             if token_not_last(key) and key.nbor(1).lower_.startswith("who"):
#                 cur = (key, *key.conjuncts)
#                 for c in cur:
#                     result.append((c, both))
#
#     return result
#
#
# def tag_label(tag: str, head_dep: str) -> str | None:
#     if (tag, head_dep) in [
#         ("NNP", "relcl"),
#         ("NNP", "acl"),
#         ("NNP", "nsubj"),
#         ("NNP", "poss"),
#         ("NNS", "poss"),
#     ]:
#         return both
#
#     return None
#
#
# def tag_base(s: Span):
#     result = list()
#
#     for token in s:
#         tag = token.tag_
#         head_dep = token.head.dep_
#         label = tag_label(tag, head_dep)
#         if label:
#             result.append((token, label))
#
#     return result
#
#
# def ner(s: Span):
#     result = list()
#
#     for ent in s.ents:
#         if ent.label_ in [
#             "PERSON",
#         ] and ent.root.dep_ in ["compound", "nsubj", "nsubjpass"]:
#             result.append((ent, both))
#         elif ent.label_ in ["ORG"] and ent.root.dep_ in ["nsubj", "nsubjpass"]:
#             result.append((ent, both))
#
#     return result
#
#
# def prep_sb(s: Span):
#     result = list()
#     for tok in s:
#         if (
#             tok.head.lower_ in ["with", "between"]
#             and tok.dep_ == "pobj"
#             and tok.pos_ in ["PROPN"]
#         ):
#             result.append((tok, both))
#
#     return result
#
#
# def acl_to(s: Span):
#     result = list()
#
#     for token in s:
#         if token.dep_ == "acl":
#             for child in token.children:
#                 if child.dep_ == "aux" and child.lower_ == "to":
#                     if token.lower_ not in [
#                         "am",
#                         "is",
#                         "are",
#                         "was",
#                         "were",
#                         "be",
#                         "has",
#                         "have",
#                         "had",
#                     ]:
#                         result.append((token, core))
#
#     return result
#
#
# def acomp_template(s: Span):
#     result = list()
#
#     for tok in s:
#         if tok.dep_ == "prep" and tok.head.lower_ in [
#             "capable",
#             "responsible",
#         ]:
#             for child in tok.children:
#                 result.append((child, core))
#     return result
#
#
# def able_to(s: Span):
#     result = list()
#
#     for tok in s:
#         if tok.dep_ == "acomp" and tok.lower_ in ["able"]:
#             for child in tok.children:
#                 result.append((child, core))
#     return result
#
#
# def nsubjpass_head(s: Span):
#     result = list()
#
#     for tok in s:
#         if tok.dep_ == "nsubjpass":
#             result.append((tok.head, core))
#     return result
#
#
# class LabelTypeException(Exception):
#     pass
#
#
# def check_bert_match():
#     with open("ie_bert_result.bin", "rb") as file:
#         results_list = pickle.load(file)
#     for i in range(10):
#         results: list[BertResult] = results_list[i]
#         for result in results:
#             pred_entities, true_entities = get_series_bio([result])
#             if set(pred_entities) != set(true_entities):
#                 loss = set(true_entities) - set(pred_entities)
#                 if loss:
#                     print(" ".join(result.tokens))
#                 for ent in loss:
#                     print(" ".join(result.tokens[ent[1] : ent[2] + 1]))
#
#
# # Input a function, then check if the function mismatch the entity
# def check_match(rule: RulePlugin):
#     data = list(
#         load_dataset("pretrained_data/2022_Kfold/intention/10/all.jsonl")
#     )
#     total_pos, total_neg = 0, 0
#     list_pos, list_neg = [], []
#
#     for i, sent, anno in data:
#         s = nlp(sent)
#         result = dispatch(
#             s[:],
#             None,
#             None,
#             funcs=(rule,),
#         )
#
#         result_chunk = list()
#         for item in result:
#             # 如有对应 chunk 则取 chunk
#             # tok = match_noun_chunk(item.token, s[:])
#             tok = item.token
#             label = item.label
#
#             # 取 Token 并转化为 Span
#             if not tok:
#                 tok = item.token
#             if isinstance(tok, Token):
#                 i = tok.i
#                 assert tok in tok.doc[i : i + 1]
#                 tok = tok.doc[i : i + 1]
#             assert isinstance(tok, Span)
#             idx = get_token_idx(tok)
#             new_fix = EntityFix(tok, idx, [], label)
#             if new_fix not in result_chunk:
#                 result_chunk.append(new_fix)
#
#         # precs = check_result_precision(s[:], result_chunk, anno)
#         # total_pos += precs
#         # total_neg += len(result_chunk) - precs
#
#     #     if len(result_chunk) != precs:
#     #         logger.warning(sent)
#     #         logger.warning(f"❌{result_chunk}")
#     #         logger.warning(anno)
#     #         for res in result_chunk:
#     #             list_neg.append(res.token.root.head.dep_)
#     #     elif len(result_chunk) != 0:
#     #         logger.warning(sent)
#     #         logger.warning(f"✅{result_chunk}")
#     #         logger.warning(anno)
#     #         for res in result_chunk:
#     #             list_pos.append(res.token.root.head.dep_)
#     #
#     # logger.error(f"✅: {total_pos}, ❌: {total_neg}")
#     # print(Counter(list_pos))
#     # print(Counter(list_neg))
#
#
# if __name__ == "__main__":
#     logging.getLogger("src").setLevel(logging.WARNING)
#     nlp: spacy.language.Language = spacy.load("en_core_web_lg")
#     check_match(nsubjpass_head)
#     # check_bert_match()
