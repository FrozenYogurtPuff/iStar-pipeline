from src.rules.entity.plugins.agent_dative_ADP import agent_dative_ADP
# Entity Rules
from src.rules.entity.plugins import *
from src.utils.typing import EntityRulePlugins


entity_plugins: EntityRulePlugins = (
    dative_PROPN, agent_dative_ADP, poss_PROPN, relcl_who, poss, dobj_pobj, word_list, ner
)
entity_autocrat: EntityRulePlugins = (relcl_who, word_list, ner)
