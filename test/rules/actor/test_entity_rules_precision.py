# import logging
# from test.rules.inspect.entity_rules import (
#     acl_to,
#     dative_propn,
#     ner,
#     prep_sb,
#     relcl_who,
#     tag_base,
# )
# from test.rules.utils.load_dataset import load_dataset
#
# import spacy
#
# from src.rules.config import (  # nopycln: import
#     actor_plugins,
#     intention_plugins,
# )
# from src.rules.dispatch import dispatch
# from src.rules.utils.seq import is_entity_type_ok
# from src.utils.spacy_utils import (
#     char_idx_to_word_idx,
#     get_token_idx,
#     match_noun_chunk,
# )
# from src.utils.typing import DatasetEntityLabel, EntityFix, Span, Token
#
# logger = logging.getLogger(__name__)
#
#
# # result = [(1, [], 'Actor')]
# # labels = [[0, 12, "Actor"], [31, 35, "Actor"], [59, 88, "Resource"]]
# def check_result_precision(
#     sent: Span, result: list[EntityFix], labels: list[DatasetEntityLabel]
# ) -> int:
#     target = 0
#     for item in result:
#         _, idx, _, attr = item
#         for label in labels:
#             begin, end, attr_hat = label
#             begin, end = char_idx_to_word_idx(sent, begin, end)
#             if begin <= idx[0] <= idx[-1] < end and is_entity_type_ok(
#                 attr, attr_hat
#             ):
#                 target += 1
#     return target
#
#
# def test_entity_rules_precision():
#     action_plugins_new = (
#         dative_propn,
#         relcl_who,
#         tag_base,
#         ner,
#         prep_sb,
#     )
#     intention_plugins_new = (acl_to,)
#     # EXCLUDE = False
#     nlp: spacy.language.Language = spacy.load("en_core_web_trf")
#     res = load_dataset("pretrained_data/2022_Kfold/intention/10/all.jsonl")
#     tp_fp, tp_fn, tp = 0, 0, 0
#     for i, sent, anno in res:
#         logger.debug(f"Before dispatch in test: {sent}")
#         s = nlp(sent)
#         result = dispatch(
#             s[:],
#             None,
#             None,
#             add_all=True,
#             noun_chunk=True,
#             funcs=intention_plugins_new,
#         )
#         result_chunk = list()
#
#         tp_fp += len(result)
#         tp_fn += len(anno)
#
#         for item in result:
#             tok = match_noun_chunk(item.token, s[:])
#             label = item.label
#             if not tok:
#                 tok = item.token
#                 if isinstance(tok, Token):
#                     i = tok.i
#                     assert tok in tok.doc[i : i + 1]
#                     tok = tok.doc[i : i + 1]
#             assert isinstance(tok, Span)
#             idx = get_token_idx(tok)
#             new_fix = EntityFix(tok, idx, [], label)
#             if new_fix not in result_chunk:
#                 result_chunk.append(new_fix)
#
#         precs = check_result_precision(s[:], result_chunk, anno)
#         tp += precs
#         if precs != len(result_chunk):
#             logger.warning(f"Sent: {s.text}")
#             logger.warning(
#                 f"Line {i}: {result_chunk}, token: {result_chunk[0][0]}"
#             )
#             logger.warning(f"current Hit {precs} out of {len(result_chunk)}")
#         else:
#             if len(result_chunk) != 0:
#                 logger.info(f"Sent: {s.text}")
#                 logger.info(f"Line {i}: {result}")
#                 logger.info(f"current Hit {precs} out of {len(result)}")
#             else:
#                 logger.debug(f"Sent: {s.text}")
#                 logger.debug(f"Line {i}: {result}")
#                 logger.debug("current Hit 0 out of 0")
#     precision = tp / tp_fp if tp_fp != 0 else 0
#     recall = tp / tp_fn if tp_fn != 0 else 0
#     logger.error(f"Total precision: {precision} about {tp}/{tp_fp}")
#     logger.error(f"Total recall: {recall} about {tp}/{tp_fn}")
#
#
# if __name__ == "__main__":
#     logging.disable(logging.WARNING)
#     test_entity_rules_precision()
