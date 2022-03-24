# Entity Rules
from src.rules.entity.resource_plugins import agent_dative_adp, poss_propn
from src.rules.entity.resource_plugins import word_list as resource_word_list
from src.rules.intention.procedure.aux_slice import (
    acl_without_to,
    agent,
    relcl,
)
from src.utils.typing import EntityRulePlugins, IntentionRuleAuxPlugins

actor_plugins: EntityRulePlugins = (
    # dative_propn,
    # relcl_who,
    # actor_tag,
    # actor_dep,
    # actor_word_list,
    # actor_ner,
    # xcomp_ask,
    # be_nsubj,
    # by_sb,
)  # TODO

resource_plugins: EntityRulePlugins = (
    # resource_dep,
    # resource_ner,
    # resource_tag,
    agent_dative_adp,
    poss_propn,
    resource_word_list,
)

intention_aux_slice_plugins: IntentionRuleAuxPlugins = (
    acl_without_to,
    relcl,
    agent,
)
