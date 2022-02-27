# Entity Rules
from src.rules.entity.plugins import (
    dative_propn,
    dobj_pobj,
    ner,
    poss,
    poss_propn,
    relcl_who,
    word_list,
)
from src.rules.entity.plugins.agent_dative_ADP import agent_dative_adp
from src.utils.typing import EntityRulePlugins

entity_plugins: EntityRulePlugins = (
    dative_propn,
    agent_dative_adp,
    poss_propn,
    relcl_who,
    poss,
    dobj_pobj,
    word_list,
    ner,
)
entity_autocrat: EntityRulePlugins = (relcl_who, word_list, ner)
