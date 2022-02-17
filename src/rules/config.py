from src.typing import EntityRulePlugins

# Entity Rules
from src.rules.entity.dative_PROPN import dative_PROPN
from src.rules.entity.dative_ADP import dative_ADP

entity_plugins: EntityRulePlugins = (
    dative_PROPN, dative_ADP
)
