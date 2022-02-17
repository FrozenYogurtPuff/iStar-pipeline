from src.typing import EntityRulePlugins

# Entity Rules
from src.rules.entity.dative_PROPN import dative_PROPN
from src.rules.entity.agent_dative_ADP import agent_dative_ADP

entity_plugins: EntityRulePlugins = (
    dative_PROPN, agent_dative_ADP
)
