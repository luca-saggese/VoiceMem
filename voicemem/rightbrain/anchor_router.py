"""Anchor Router: converte l'input utente e i segnali del Preprocessor in MemoryQueryPlan.

Logica principale:
  1. Riutilizza la corrispondenza entity di CognitiveGraphStore del cervello sinistro, senza chiamare LLM aggiuntivo
  2. entity.entity_type → anchor_type (persona → person, progetto → project…)
  3. entity_edges → anchor entity_edge (il compito dato dal Boss è più preciso del Boss stesso)
  4. Nessun match → aggiunge fallback user_self + global_style

Non dipende dalle tabelle del cervello destro, legge solo SQLite del cervello sinistro.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .types import CurrentSignals, MemoryAnchor, MemoryQueryPlan

if TYPE_CHECKING:
    from voicemem.leftbrain.cognitive_graph.store import CognitiveGraphStore

# EntityType.value del cervello sinistro → anchor_type MemoryAnchor
# Originariamente qui si mappava secondo SlotType del cervello sinistro (una tassonomia di 7 categorie entity-indipendente da SlotV2);
# SlotType è stato eliminato con l'unificazione della tassonomia slot (ora usa SlotV2 per rappresentare "domini della vita"),
# e ciò che serve davvero qui è sempre stato "che tipo di entità è questo" — EntityType
# lo esprime direttamente, non serve più un livello di traduzione tramite slot.
# organization fuso in person (in ENTITY_TYPE_TO_SLOT originale ORGANIZATION era anch'esso fuso nello
# slot people); event fuso in knowledge (la mappa originale assegnava anche EVENT a knowledge).
# Le entità di tipo user/preference non sono nella tabella, usano il default .get(..., "knowledge") qui sotto,
# coerente con il comportamento di fallback del percorso SlotType originale.
_ENTITY_TYPE_TO_ANCHOR: dict[str, str] = {
    "person":       "person",
    "organization": "person",
    "project":      "project",
    "task":         "task",
    "knowledge":    "knowledge",
    "event":        "knowledge",
    "place":        "place",
    "routine":      "routine",
    "asset":        "asset",
}

# anchor_type → valore default role
_ANCHOR_ROLE: dict[str, str] = {
    "person":       "subject",
    "user":         "global_profile",
    "project":      "context",
    "task":         "topic",
    "knowledge":    "context",
    "place":        "context",
    "routine":      "context",
    "asset":        "context",
    "entity_edge":  "trigger",
    "global_style": "global_profile",
}

# anchor_type → peso di retrieval
_ANCHOR_WEIGHT: dict[str, float] = {
    "task":         1.0,
    "entity_edge":  0.9,
    "project":      0.9,
    "person":       0.7,
    "knowledge":    0.6,
    "place":        0.5,
    "routine":      0.5,
    "asset":        0.4,
    "user":         0.3,
    "global_style": 0.3,
}

_CANONICAL_EMOTIONS = {"ansia", "tristezza", "ingiustizia", "solitudine", "confusione", "pace", "gioia", "stanchezza"}

_EMOTION_KEYWORDS: list[tuple[str, str]] = [
    # ansia
    ("ansia", "ansia"), ("stress", "ansia"), ("tensione", "ansia"), ("preoccupazione", "ansia"),
    ("paura", "ansia"), ("spaventato", "ansia"), ("inquietudine", "ansia"), ("panico", "ansia"),
    # tristezza
    ("tristezza", "tristezza"), ("dispiacere", "tristezza"), ("perdita", "tristezza"), ("depressione", "tristezza"),
    ("addolorato", "tristezza"), ("disperazione", "tristezza"), ("crollo", "tristezza"), ("malessere", "tristezza"),
    # ingiustizia/arrabbiatura
    # "arrabbiato/furioso/irritato" prima non erano nella tabella, mentre "stress" era nel gruppo ansia — quindi "sono così arrabbiato,
    # il mio capo mi stressa sempre" veniva classificato come [ansia], diceva chiaramente arrabbiato ma non lo riconosceva. La tabella corrisponde in ordine,
    # quindi questi devono venire prima di parole generiche come "stress" per sovrascriverle (vedi uso di _EMOTION_KEYWORDS qui sotto).
    ("ingiustizia", "ingiustizia"), ("rabbia", "ingiustizia"), ("indignazione", "ingiustizia"), ("insoddisfazione", "ingiustizia"),
    ("arrabbiato", "ingiustizia"), ("furioso", "ingiustizia"), ("infuriato", "ingiustizia"), ("irritato", "ingiustizia"),
    ("sopraffare", "ingiustizia"), ("ingiusto", "ingiustizia"),
    # solitudine
    ("solitudine", "solitudine"), ("vuoto", "solitudine"), ("solitario", "solitudine"),
    # confusione
    ("confusione", "confusione"), ("contraddizione", "confusione"), ("smarrimento", "confusione"), ("esitazione", "confusione"),
    # pace
    ("pace", "pace"), ("distacco", "pace"), ("calma", "pace"), ("accettazione", "pace"),
    ("serenità", "pace"),
    # gioia
    ("gioia", "gioia"), ("felicità", "gioia"), ("eccitazione", "gioia"), ("aspettativa", "gioia"),
    ("piacere", "gioia"), ("soddisfazione", "gioia"), ("orgoglio", "gioia"), ("aspirazione", "gioia"),
    ("gratitudine", "gioia"), ("leggerezza", "gioia"), ("determinazione", "gioia"),
    # stanchezza
    ("stanchezza", "stanchezza"), ("stanco", "stanchezza"), ("sonnolenza", "stanchezza"), ("impotenza", "stanchezza"),
    ("esausto", "stanchezza"),
]

# Versione inglese — VoiceMem supporta sia cinese che inglese, i tag emotion non possono riconoscere solo parole chiave cinesi,
# altrimenti in scenari puramente inglesi (es. emotion_tag generati da ASR/TTS in inglese) verrebbero tutti classificati erroneamente come "pace".
_EMOTION_KEYWORDS_EN: list[tuple[str, str]] = [
    # ansia
    ("anxious", "ansia"), ("anxiety", "ansia"), ("nervous", "ansia"), ("worried", "ansia"),
    ("worry", "ansia"), ("stressed", "ansia"), ("stress", "ansia"), ("tense", "ansia"),
    ("fearful", "ansia"), ("afraid", "ansia"), ("scared", "ansia"), ("panicked", "ansia"),
    ("panic", "ansia"), ("uneasy", "ansia"), ("apprehensive", "ansia"),
    # tristezza
    ("sad", "tristezza"), ("sadness", "tristezza"), ("upset", "tristezza"), ("depressed", "tristezza"),
    ("disappointed", "tristezza"), ("heartbroken", "tristezza"), ("miserable", "tristezza"),
    ("dejected", "tristezza"), ("despair", "tristezza"), ("sorrowful", "tristezza"), ("grief", "tristezza"),
    # ingiustizia / rabbia
    ("wronged", "ingiustizia"), ("angry", "ingiustizia"), ("anger", "ingiustizia"), ("mad", "ingiustizia"),
    ("furious", "ingiustizia"), ("irritated", "ingiustizia"), ("annoyed", "ingiustizia"), ("frustrated", "ingiustizia"),
    ("resentful", "ingiustizia"), ("indignant", "ingiustizia"), ("unfair", "ingiustizia"), ("bitter", "ingiustizia"),
    # solitudine
    ("lonely", "solitudine"), ("loneliness", "solitudine"), ("isolated", "solitudine"), ("empty", "solitudine"),
    ("alone", "solitudine"),
    # confusione
    ("conflicted", "confusione"), ("torn", "confusione"), ("confused", "confusione"), ("uncertain", "confusione"),
    ("hesitant", "confusione"), ("ambivalent", "confusione"), ("indecisive", "confusione"), ("perplexed", "confusione"),
    ("lost", "confusione"),
    # pace
    ("calm", "pace"), ("relaxed", "pace"), ("peaceful", "pace"), ("composed", "pace"),
    ("serene", "pace"), ("settled", "pace"), ("neutral", "pace"),
    # gioia
    ("happy", "gioia"), ("happiness", "gioia"), ("joy", "gioia"), ("joyful", "gioia"),
    ("excited", "gioia"), ("excitement", "gioia"), ("glad", "gioia"), ("pleased", "gioia"),
    ("delighted", "gioia"), ("proud", "gioia"), ("grateful", "gioia"), ("thankful", "gioia"),
    ("relieved", "gioia"), ("hopeful", "gioia"), ("cheerful", "gioia"), ("satisfied", "gioia"),
    ("content", "gioia"), ("amused", "gioia"),
    # stanchezza
    ("tired", "stanchezza"), ("exhausted", "stanchezza"), ("fatigue", "stanchezza"), ("fatigued", "stanchezza"),
    ("weary", "stanchezza"), ("drained", "stanchezza"), ("sleepy", "stanchezza"), ("worn out", "stanchezza"),
]

_EN_KEYWORD_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), canonical)
    for kw, canonical in _EMOTION_KEYWORDS_EN
]


def normalize_emotion_strict(emotion: str) -> str | None:
    """Mappa una stringa di emozione libera (cinese o inglese) a un'etichetta canonica;
    restituisce None quando non corrisponde nulla.

    I chiamanti legati agli anchor dovrebbero usare questa versione: le parole emotive non riconosciute (guilty / jealous /
    nostalgic…) prima venivano tutte fallbackate su "pace", poi scritte negli anchor di retrieval con il peso più alto (1.2) —
    equivale a iniettare un segnale errato ad alto peso nella query. Se non riconosci, non aggiungere l'anchor emotion,
    è meglio aggiungere uno sbagliato."""
    e = emotion.strip()
    if not e:
        return None
    if e in _CANONICAL_EMOTIONS:
        return e
    # Prendi il primo per **posizione della parola nella frase**, non per ordine della tabella.
    # L'ordine della tabella è scritto per gruppi (gruppo ansia prima, gruppo ingiustizia dopo), se si corrisponde secondo l'ordine della tabella
    # "sono così arrabbiato, il mio capo mi stressa sempre" colpirebbe prima "stress" → [ansia], mentre la prima parola detta dalla persona
    # è proprio "arrabbiato". La parola emotiva detta per prima durante il parlato è solitamente l'emozione principale.
    best = None
    for keyword, canonical in _EMOTION_KEYWORDS:
        i = e.find(keyword)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, canonical)
    if best is not None:
        return best[1]
    for pattern, canonical in _EN_KEYWORD_RE:
        if pattern.search(e):
            return canonical
    return None


def normalize_emotion(emotion: str) -> str:
    """Come normalize_emotion_strict, ma fa fallback su pace per i chiamanti che
    hanno bisogno di un'etichetta canonica garantita (es. denominazione nodo grafo)."""
    return normalize_emotion_strict(emotion) or "pace"


_STOP = {
    "who", "is", "are", "was", "were", "what", "when", "where", "why",
    "how", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "my", "your", "his", "her", "their", "our", "tell",
    "me", "about", "did", "do", "does", "has", "have", "had", "can",
    "could", "would", "should", "with", "from", "that", "this", "it",
    "be", "been", "being", "not", "no", "any", "some", "which",
}


class AnchorRouter:
    """Genera MemoryQueryPlan dall'input corrente.

    cognitive_store può essere None (in questo caso restituisce solo anchor fallback).
    """

    def __init__(
        self,
        cognitive_store: "CognitiveGraphStore | None" = None,
    ) -> None:
        self._store = cognitive_store

    def build_query_plan(
        self,
        query: str,
        user_id: str,
        *,
        signals: CurrentSignals | None = None,
        entities: list[str] | None = None,
        emotion: str | None = None,
        context: str | None = None,
    ) -> MemoryQueryPlan:
        """``context``: frase precedente dell'agent. Questa frase dell'utente spesso deve essere inserita lì dentro per essere completa ("allora lasciamo stare"),
        quindi le entità che menziona entrano anche negli anchor, ma degradate al ruolo context, peso dimezzato — è sfondo,
        non il soggetto principale di questo turno dell'utente. clean_text rimane solo l'originale dell'utente."""
        anchors = self._build_anchors(
            query, user_id, hint_entities=entities, hint_emotion=emotion,
            context_text=context,
        )
        return MemoryQueryPlan(
            user_id=user_id,
            clean_text=query.strip(),
            anchors=anchors,
            current_signals=signals or CurrentSignals(),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    # Le entità corrispondenti nella frase dell'agent subiscono questa riduzione del peso dell'anchor (sfondo < soggetto principale di questo turno dell'utente)
    _CONTEXT_WEIGHT_SCALE = 0.5

    def _build_anchors(self, query: str, user_id: str, hint_entities: list[str] | None = None,
                       hint_emotion: str | None = None,
                       context_text: str | None = None) -> list[MemoryAnchor]:
        anchors: list[MemoryAnchor] = []
        seen_ids: set[str] = set()

        if self._store is not None:
            from voicemem.leftbrain.cognitive_graph.store import normalize_name

            # Prima scansiona la frase dell'utente: le entità menzionate da entrambi i lati vengono prima occupate con peso pieno, non verranno abbassate dalla passata sfondo
            sources: list[tuple[str, float]] = [(query, 1.0)]
            if context_text and context_text.strip():
                sources.append((context_text, self._CONTEXT_WEIGHT_SCALE))

            matched_entities: list[tuple[Any, float]] = []
            all_ents = self._store.find_entities(user_id)   # per reverse lookup cinese, le due passate condividono una query
            for text, scale in sources:
                # Parole candidate: parole inglesi + bigram (logica originale)
                raw_words = re.findall(r"\b\w{2,}\b", text.lower())
                candidates = [w for w in raw_words if w not in _STOP]
                for i in range(len(raw_words) - 1):
                    a, b = raw_words[i], raw_words[i + 1]
                    if a not in _STOP and b not in _STOP:
                        candidates.append(f"{a} {b}")

                for cand in candidates:
                    ents = self._store.find_entities_by_name_fuzzy(user_id, cand)
                    for e in ents:
                        if e.id not in seen_ids:
                            seen_ids.add(e.id)
                            matched_entities.append((e, scale))

                # Reverse lookup cinese: scorri tutti i nomi entity, controlla se appaiono nell'input
                # (il cinese non ha spazi per il word splitting, le regex di confine parola inglese non funzionano per il cinese)
                text_lower = text.lower()
                for e in all_ents:
                    if e.id in seen_ids:
                        continue
                    name_l = e.name.lower()
                    name_n = (e.name_norm or "").lower()
                    if len(name_l) >= 2 and (name_l in text_lower or name_n in text_lower):
                        seen_ids.add(e.id)
                        matched_entities.append((e, scale))

            # Entity fornite dal modulo voce: usa direttamente il nome come anchor (coerente con la scrittura Ingest)
            if hint_entities:
                for name in hint_entities:
                    key = name.lower().strip()
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    anchors.append(MemoryAnchor(
                        anchor_type="entity",
                        anchor_id=key,
                        role="subject",
                        weight=1.0,
                        confidence=1.0,
                    ))

            for ent, scale in matched_entities:
                anchor_type = _ENTITY_TYPE_TO_ANCHOR.get(ent.entity_type.value, "knowledge")
                anchors.append(MemoryAnchor(
                    anchor_type=anchor_type,
                    anchor_id=ent.id,
                    # Appare solo nella frase dell'agent: registra come context, peso dimezzato
                    role=("context" if scale < 1.0
                          else _ANCHOR_ROLE.get(anchor_type, "context")),
                    weight=_ANCHOR_WEIGHT.get(anchor_type, 0.5) * scale,
                    confidence=ent.confidence,
                ))
                # Quando il cervello destro scrive, l'anchor entity usa name.lower().strip() (scrittura entity anchor_id in core.py::Ingest),
                # non l'entity.id del cervello sinistro — due sistemi ID non comunicanti.
                # Qui si aggiunge extra un anchor "entity" normalizzato con la stessa regola,
                # così le entità trovate per corrispondenza fuzzy possono anche trovare l'anchor appeso durante la scrittura originale.
                name_key = ent.name.lower().strip()
                if name_key not in seen_ids:
                    seen_ids.add(name_key)
                    anchors.append(MemoryAnchor(
                        anchor_type="entity",
                        anchor_id=name_key,
                        role="subject",
                        weight=1.0 * scale,
                        confidence=ent.confidence,
                    ))

            # Anchor entity_edge: quando ≥2 entità vengono trovate, aggiungi anche gli archi tra loro
            if len(matched_entities) >= 2:
                entity_ids = [e.id for e, _ in matched_entities]
                for e, _scale in matched_entities:
                    edges = self._store.edges_for_entity(e.id, user_id)
                    for edge in edges:
                        if (edge.from_entity_id in entity_ids
                                and edge.to_entity_id in entity_ids
                                and edge.id not in seen_ids):
                            seen_ids.add(edge.id)
                            anchors.append(MemoryAnchor(
                                anchor_type="entity_edge",
                                anchor_id=edge.id,
                                role="trigger",
                                weight=_ANCHOR_WEIGHT["entity_edge"],
                                confidence=edge.confidence,
                            ))

        # Anchor emotion: retrieval di eventi emotivi passati basati sull'emozione corrente (peso più alto).
        # Versione strict: parole emotive non riconosciute non aggiungono anchor (invece di fallback su "pace").
        if hint_emotion:
            canonical = normalize_emotion_strict(hint_emotion)
            if canonical is not None:
                anchors.append(MemoryAnchor(
                    anchor_type="emotion",
                    anchor_id=canonical,
                    role="trigger",
                    weight=1.2,
                    confidence=1.0,
                ))

        # Fallback: user_self + global_style sempre aggiunti (peso basso)
        anchors.append(MemoryAnchor(
            anchor_type="user",
            anchor_id="user_self",
            role="global_profile",
            weight=_ANCHOR_WEIGHT["user"],
            confidence=1.0,
        ))
        anchors.append(MemoryAnchor(
            anchor_type="global_style",
            anchor_id="global_style",
            role="global_profile",
            weight=_ANCHOR_WEIGHT["global_style"],
            confidence=1.0,
        ))

        return anchors
