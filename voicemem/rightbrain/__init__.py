"""Experience Layer del cervello destro.

Due livelli:

  · **Livello memoria** heartnote + response_experience, retrieval per anchor (store/experience_repository)
  · **Livello giudizio** rb_traits + rb_evidence, un nodo = un giudizio su questa persona,
    claim con vettore, retrieval semantico per query, restituito come rb_hit con source="profile"
    (vedi traits_store.py e brain._rb_trait_hits)

Il livello giudizio ha sostituito il vecchio grafo slot→entity→heartnote. In quella struttura il livello entity svolgeva tre ruoli
(giudizio / argomento / parola emotiva), nei test reali si è trasformato in un miscuglio tipo "tristezza ×61", "Jiaqi ×52",
le descrizioni dovevano essere integrate con consolidamento batch post-hoc. Le tabelle vecchie (rb_slots/rb_entities) sono mantenute in sola lettura in una versione,
non si scrive più.
"""
from .anchor_router import AnchorRouter
from .attribution_manager import AttributionManager
from .brain import RightBrain, RightBrainHit
from .experience_repository import ExperienceRepository
from .graph_store import RBEntity, RBSlot, RightBrainGraphStore
from .store import RightBrainStore
from .types import (
    CurrentSignals,
    MemoryAnchor,
    MemoryQueryPlan,
    RightBrainContext,
    RightBrainMemory,
)

__all__ = [
    "AnchorRouter",
    "AttributionManager",
    "RightBrain",
    "RightBrainHit",
    "ExperienceRepository",
    "RightBrainGraphStore",
    "RBSlot",
    "RBEntity",
    "RightBrainStore",
    "CurrentSignals",
    "MemoryAnchor",
    "MemoryQueryPlan",
    "RightBrainContext",
    "RightBrainMemory",
]
