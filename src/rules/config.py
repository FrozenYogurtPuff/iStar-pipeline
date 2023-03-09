# Entity Rules
from src.rules.entity.actor_plugins.include import relcl_who  # nopycln: import
from src.rules.entity.actor_plugins.include import xcomp_ask  # nopycln: import
from src.rules.entity.actor_plugins.include import (  # nopycln: import
    be_nsubj,
    by_sb,
    dative_propn,
)
from src.rules.entity.actor_plugins.include import (
    dep as actor_dep,  # nopycln: import
)
from src.rules.entity.actor_plugins.include import (
    ner as actor_ner,  # nopycln: import
)
from src.rules.entity.actor_plugins.include import (
    tag as actor_tag,  # nopycln: import
)

# from src.rules.entity.actor_plugins.include import (
#     word_list as actor_word_list,  # nopycln: import
# )
from src.rules.entity.resource_plugins import (  # nopycln: import
    agent_dative_adp,
    poss_propn,
)
from src.rules.entity.resource_plugins import (
    word_list as resource_word_list,  # nopycln: import
)
from src.rules.intention.aux_slice import (
    acl_without_to as awt_slice,  # nopycln: import
)
from src.rules.intention.aux_slice import agent  # nopycln: import
from src.rules.intention.aux_slice import (
    relcl as relcl_slice,  # nopycln: import
)
from src.rules.intention.intention_plugins import (  # nopycln: import
    acl_to,
    acl_without_to,
    advcl,
    pcomp_ing,
    relcl,
    xcomp_to,
)
from src.utils.typing import IntentionAuxPlugins, RulePlugins

actor_plugins: RulePlugins = (
    # dative_propn,
    # relcl_who,
    actor_tag,
    actor_dep,
    # actor_word_list,
    actor_ner,
    # xcomp_ask,
    be_nsubj,
    # by_sb,
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
    # xcomp_to,
    acl_to,
    # acl_without_to,
    # relcl,
    # pcomp_ing,
    # advcl,
)
