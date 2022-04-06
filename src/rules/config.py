# Entity Rules
from src.rules.entity.actor_plugins import (
    be_nsubj,
    by_sb,
    dative_propn,
    relcl_who,
    xcomp_ask,
)
from src.rules.entity.resource_plugins import agent_dative_adp, poss_propn
from src.rules.entity.resource_plugins import word_list as resource_word_list
from src.rules.intention.aux_slice import acl_without_to as awt_slice
from src.rules.intention.aux_slice import agent
from src.rules.intention.aux_slice import relcl as relcl_slice
from src.rules.intention.intention_plugins import acl_to, xcomp_to
from src.utils.typing import IntentionAuxPlugins, RulePlugins

actor_plugins: RulePlugins = (
    dative_propn,
    relcl_who,
    # actor_tag,
    # actor_dep,
    # actor_word_list,
    # actor_ner,
    xcomp_ask,
    be_nsubj,
    by_sb,
)

resource_plugins: RulePlugins = (
    # resource_dep,
    # resource_ner,
    # resource_tag,
    agent_dative_adp,
    poss_propn,
    resource_word_list,
)

intention_aux_slice_plugins: IntentionAuxPlugins = (
    awt_slice,
    relcl_slice,
    agent,
)

intention_plugins: RulePlugins = (
    xcomp_to,
    acl_to,
    # acl_without_to,
    # relcl,
    # pcomp_ing,
    # advcl,
)
# TODO: 将单词映射为动词短语
