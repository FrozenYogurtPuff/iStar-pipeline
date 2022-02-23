from src.utils.typing import EntityRulePlugins

# Entity Rules
from src.rules.entity.dative_PROPN import dative_PROPN
from src.rules.entity.agent_dative_ADP import agent_dative_ADP
from src.rules.entity.poss_PROPN import poss_PROPN
from src.rules.entity.relcl_who import relcl_who
# from src.rules.entity.nsubj import nsubj
# from src.rules.entity.attr import attr
from src.rules.entity.poss import poss
from src.rules.entity.dobj_pobj import dobj_pobj
from src.rules.entity.word_list import word_list
from src.rules.entity.ner import ner


# TODO: 可以让word_list的结果override其他结果吗
entity_plugins: EntityRulePlugins = (
    dative_PROPN, agent_dative_ADP, poss_PROPN, relcl_who, poss, dobj_pobj, word_list, ner
)
