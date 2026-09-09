"""Fusione corpo calloso: retrieval dual-channel e tipi prompt risposta."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voicemem.utils.audio.emotion.types import EmotionAttribution, VAD


@dataclass(frozen=True)
class LeftMemoryHit:
    memory_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RightMemoryHit:
    turn_id: str
    analysis_text: str
    vad: VAD
    retrieval_snippet: list[str] = field(default_factory=list)
    left_context_summary: str | None = None


@dataclass
class FusionRetrievalResult:
    asr_text: str
    left_hits: list[LeftMemoryHit] = field(default_factory=list)
    right_hits: list[RightMemoryHit] = field(default_factory=list)
    left_graph_appendix: str = ""


@dataclass
class ReplyRetrievalBundle:
    """Risultati retrieval memoria risposta (per uso build_reply_context_prompt)."""

    retrieval: FusionRetrievalResult
    graph_emotion_context: str = ""


@dataclass
class ReplyContextPrompt:
    """Contesto risposta per分区: blocco semantico + blocco emotivo + placeholder Persona ecc."""

    semantic_block: str
    emotional_block: str
    left_context_summary: str
    persona_block: str = ""
    acoustic_block: str = ""
    graph_emotion_block: str = ""
    current_attribution_block: str = ""
    system_block: str = ""
    user_block: str = ""


@dataclass
class ReplyContextBundle:
    """Memoria risposta post-retrieval + assemblaggio."""

    retrieval: ReplyRetrievalBundle
    prompt: ReplyContextPrompt


@dataclass
class AnomalyTurnResult:
    retrieval: FusionRetrievalResult
    #: Contesto reply prima dell'attribuzione (per Omni); risposta finale usa ``TurnProcessResult.reply_context``.
    reply_prompt: ReplyContextPrompt
    attribution: EmotionAttribution
    pre_attribution_reply: ReplyContextPrompt | None = None

    def __post_init__(self) -> None:
        if self.pre_attribution_reply is None:
            self.pre_attribution_reply = self.reply_prompt
