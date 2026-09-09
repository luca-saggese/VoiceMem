"""Tipi di dati del livello Experience del cervello destro.

Tre classi principali di memoria:
  user_interaction_profile   — Preferenze stilistiche a lungo termine dell'utente
  heartnote                   — Pattern emotivi contestuali
  response_experience         — Esperienze di risposta riuscite/sbagliate (massima priorità)

Punto di ingresso retrieval: MemoryAnchor → aggancia entity del cervello sinistro
Piano query: MemoryQueryPlan → segnali Preprocessor + anchors
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Anchor ────────────────────────────────────────────────────────────────────

AnchorType = Literal[
    "user", "person", "project", "task", "knowledge",
    "place", "routine", "asset", "event",
    "entity_edge", "slot", "current_session", "global_style",
]

AnchorRole = Literal[
    "topic", "subject", "object", "context",
    "trigger", "relationship", "evidence", "global_profile",
]

MemoryClass = Literal[
    "heartnote",
    "response_experience",
]

TTL = Literal["session", "short_term", "long_term"]


@dataclass
class MemoryAnchor:
    anchor_type: AnchorType
    anchor_id: str | None        # None = globale (es. user_self, global_style)
    role: AnchorRole
    weight: float = 1.0
    confidence: float = 1.0


# ── Query Plan ────────────────────────────────────────────────────────────────

@dataclass
class CurrentSignals:
    """Segnali rilevati dal Preprocessor nel turno corrente, non memorizzati, usati solo per priorità di retrieval."""
    affect_hint: str | None = None                          # "frustrated" / "satisfied" ...
    affect_intensity: Literal["low", "medium", "high"] | None = None
    dissatisfaction_signal: bool = False
    correction_signal: bool = False
    short_answer_needed: bool = False


@dataclass
class MemoryQueryPlan:
    user_id: str
    clean_text: str
    anchors: list[MemoryAnchor] = field(default_factory=list)
    current_signals: CurrentSignals = field(default_factory=CurrentSignals)


# ── Right Brain Memory ────────────────────────────────────────────────────────

@dataclass
class RightBrainMemory:
    id: str
    user_id: str
    memory_class: MemoryClass
    content: str
    condition: str | None           # Descrizione contesto applicabile (opzionale)
    priority: float                 # 0~1, response_experience solitamente la più alta
    confidence: float
    ttl: TTL
    metadata: dict[str, Any]        # Motivo fallimento, next_time_policy, ecc.
    evidence_turn_ids: list[str]
    evidence_memory_ids: list[str]
    created_at: str
    updated_at: str
    #: Punteggio anchor trovato in questa ricerca (SUM(link.weight*link.confidence), vedi
    #: store.search_by_anchors). Non salvato nel database — la stessa memoria ha punteggi diversi per query diverse,
    #: trasportata solo nei risultati di retrieval, utile al sorting finale per mescolare "rilevanza retrieval" nella priority statica.
    anchor_score: float = 0.0


@dataclass
class RightBrainAnchorLink:
    id: str
    user_id: str
    right_memory_id: str
    anchor_type: AnchorType
    anchor_id: str | None
    role: AnchorRole
    weight: float
    confidence: float
    created_at: str


# ── Retrieval Result ──────────────────────────────────────────────────────────

@dataclass
class RightBrainContext:
    """Risultati di retrieval del cervello destro, per uso del Prompt Builder.

    Contiene solo heartnote e response_experience.
    user_interaction_profile è gestito indipendentemente dal livello pre-stimolo (UserProfileStore).
    """
    response_experiences: list[RightBrainMemory] = field(default_factory=list)
    situation_patterns: list[RightBrainMemory] = field(default_factory=list)
    current_signals: CurrentSignals = field(default_factory=CurrentSignals)

    def is_empty(self) -> bool:
        return not (self.response_experiences or self.situation_patterns)

    def to_prompt_block(self) -> str:
        """Genera blocco testuale da iniettare nel prompt LLM.

        Ogni memoria ha un prefisso data [YYYY-MM-DD] — senza data, il modello downstream non può assolutamente determinare "quando è successo"
        per domande di tipo (ragionamento temporale); la data esiste sempre in created_at,
        è stata solo persa durante il rendering precedente."
        """
        lines: list[str] = []

        def _date(m: RightBrainMemory) -> str:
            d = (m.created_at or "")[:10]
            return f"[{d}] " if d else ""

        if self.response_experiences:
            lines.append("[Response experience]")
            for m in self.response_experiences:
                cond = f" (when: {m.condition})" if m.condition else ""
                lines.append(f"- {_date(m)}{m.content}{cond}")

        if self.situation_patterns:
            lines.append("[Situation pattern]")
            for m in self.situation_patterns:
                cond = f" (when: {m.condition})" if m.condition else ""
                lines.append(f"- {_date(m)}{m.content}{cond}")

        sigs = self.current_signals
        hints: list[str] = []
        if sigs.dissatisfaction_signal:
            hints.append("user shows dissatisfaction this turn")
        if sigs.correction_signal:
            hints.append("user is correcting assistant")
        if sigs.affect_hint:
            intensity = f" ({sigs.affect_intensity})" if sigs.affect_intensity else ""
            hints.append(f"affect={sigs.affect_hint}{intensity}")
        if hints:
            lines.append(f"[Current signals: {'; '.join(hints)}]")

        return "\n".join(lines)
