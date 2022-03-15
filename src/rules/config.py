# Entity Rules
from src.rules.entity.plugins import (
    agent_dative_adp,
    dative_propn,
    dep,
    ner,
    poss_propn,
    relcl_who,
    tag,
)
from src.rules.intention.procedure.aux_slice import (
    acl_without_to,
    agent,
    relcl,
)
from src.utils.typing import EntityRulePlugins, IntentionRuleAuxPlugins

entity_plugins: EntityRulePlugins = (
    dative_propn,
    agent_dative_adp,
    poss_propn,
    relcl_who,
    tag,
    dep,
    # word_list,
    ner,
)

intention_aux_slice_plugins: IntentionRuleAuxPlugins = (
    acl_without_to,
    relcl,
    agent,
)
