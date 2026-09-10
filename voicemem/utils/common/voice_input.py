"""voice_input.py - Input adapter for the voice module.

Converts structured voice-module output into the VoiceMem left-brain format.

Voice-module output format:
{
  "id": "str",
  "time_stamp": {"begin": "str", "end": "str"},
  "slots": ["str", ...],
  "contents": [
    {
      "sub_id":        "str",
      "time_start":    "str",
      "time_end":      "str",
      "sentence":      "str",
      "voiceprint_id": "str",
    "emotion":       "str"   # emotion2vec output; may be empty
    }
  ]
}

The voiceprint_id -> name/entity_id mapping is managed by VoiceprintRegistry.
"""
from __future__ import annotations

import os as _os

import os

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class VoiceContent:
    sub_id: str
    time_start: str
    time_end: str
    sentence: str
    voiceprint_id: str
    emotion: str = ""          # emotion2vec label, for example neutral/anxious

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceContent":
        return cls(
            sub_id=str(d.get("sub_id", "")),
            time_start=str(d.get("time_start", "")),
            time_end=str(d.get("time_end", "")),
            sentence=str(d.get("sentence", "")).strip(),
            voiceprint_id=str(d.get("voiceprint_id", "")),
            emotion=str(d.get("emotion", "") or ""),
        )


@dataclass
class VoiceInput:
    id: str
    time_stamp: dict          # {"begin": str, "end": str}
    slots: list[str]
    contents: list[VoiceContent]
    environment: str = ""     # AST/CLAP background sound description, e.g. "background sounds: Washing machine(0.82)"
    #: The agent side; contents contains the user side. The extractor uses it to disambiguate.
    agent_reply: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceInput":
        ts = d.get("time_stamp") or {}
        if isinstance(ts, str):
            ts = {"begin": ts, "end": ts}
        return cls(
            id=str(d.get("id", "")),
            time_stamp=ts,
            slots=[str(s) for s in (d.get("slots") or [])],
            contents=[VoiceContent.from_dict(c) for c in (d.get("contents") or [])],
            agent_reply=str(d.get("agent_reply", "") or ""),
        )

    @property
    def begin_time(self) -> str:
        return self.time_stamp.get("begin", "")

    @property
    def end_time(self) -> str:
        return self.time_stamp.get("end", "")

    def full_transcript(self) -> str:
        return " ".join(c.sentence for c in self.contents if c.sentence)

    def dominant_emotion(self) -> str:
        """Return the most frequent non-empty, non-neutral emotion."""
        from collections import Counter
        neutral = {"-", "中性", "neutral", "unknown", "未知", ""}
        emos = [c.emotion for c in self.contents if c.emotion and c.emotion not in neutral]
        if not emos:
            return ""
        return Counter(emos).most_common(1)[0][0]


# ── Voiceprint registry ──────────────────────────────────────────────────────

@dataclass
class VoiceprintEntry:
    """Registration information for one voiceprint."""
    role: str         # "user" | "assistant"
    name: str         # Display name; defaults to voiceprint_id when unset.
    entity_id: str    # ID in the left-brain cognitive_graph entities table.


class VoiceprintRegistry:
    """Persistent voiceprint_id -> name + entity_id registry.

        Core behavior:
            - Unknown voiceprints default to role="user" for multi-person conversations.
            - bind() associates a voiceprint_id with a known name and/or entity_id.
            - voice_input_to_messages() replaces "Speaker N" with the real name.
            - The extractor can then link memories to that person naturally.
            - entity_id is stored in memory metadata for direct future lookup.
    """

    ROLE_USER      = "user"
    ROLE_ASSISTANT = "assistant"

    def __init__(self, registry_path: Path, entity_resolver: Any = None) -> None:
        self._path = registry_path
        self._entries: dict[str, VoiceprintEntry] = {}
        #: Name -> cognitive-graph entity_id, injected by the orchestrator.
        self._resolve_entity = entity_resolver
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for vpid, v in data.items():
                if isinstance(v, dict):
                    self._entries[vpid] = VoiceprintEntry(
                        role=v.get("role", self.ROLE_USER),
                        name=v.get("name", vpid),
                        entity_id=v.get("entity_id", ""),
                    )
                else:
                    # Support the legacy format containing only a role string.
                    self._entries[vpid] = VoiceprintEntry(
                        role=str(v), name=vpid, entity_id=""
                    )
        except Exception:
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            vpid: {"role": e.role, "name": e.name, "entity_id": e.entity_id}
            for vpid, e in self._entries.items()
        }
        self._path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ── Public API ─────────────────────────────────────────────────────────────

    def bind(
        self,
        voiceprint_id: str,
        *,
        name: str | None = None,
        entity_id: str | None = None,
        role: str = ROLE_USER,
    ) -> VoiceprintEntry:
        """Bind a voiceprint to a name and/or entity_id incrementally."""
        if role not in (self.ROLE_USER, self.ROLE_ASSISTANT):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        existing = self._entries.get(voiceprint_id)
        entry = VoiceprintEntry(
            role=role,
            name=name or (existing.name if existing else voiceprint_id),
            entity_id=entity_id or (existing.entity_id if existing else ""),
        )
        self._entries[voiceprint_id] = entry
        self._save()
        return entry

    # Backward-compatible interface.
    def register(self, voiceprint_id: str, role: str, name: str | None = None) -> None:
        self.bind(voiceprint_id, role=role, name=name)

    def get(self, voiceprint_id: str) -> VoiceprintEntry:
        """Return registration info; unknown voiceprints get a default user entry."""
        return self._entries.get(
            voiceprint_id,
            VoiceprintEntry(role=self.ROLE_USER, name=voiceprint_id, entity_id=""),
        )

    def resolve(self, voiceprint_id: str) -> str:
        return self.get(voiceprint_id).role

    def display_name(self, voiceprint_id: str) -> str:
        return self.get(voiceprint_id).name

    def entity_id(self, voiceprint_id: str) -> str:
        """Return the bound entity, lazily resolving it by name when needed.

        Lazy resolution is needed because self-introductions happen before the
        extractor creates the entity; resolving during bind would usually miss it.
        """
        entry = self.get(voiceprint_id)
        if entry.entity_id or not self._resolve_entity:
            return entry.entity_id
        # Only resolve registered entries. An unregistered voiceprint gets a
        # temporary entry whose name is the ID, not a meaningful person name.
        # If the graph does not contain the entity yet, retry on the next call.
        if voiceprint_id not in self._entries or not entry.name:
            return ""
        try:
            eid = self._resolve_entity(entry.name) or ""
        except Exception:
            return ""
        if eid:
            self.bind(voiceprint_id, name=entry.name, entity_id=eid, role=entry.role)
        return eid

    def all_display_names(self) -> list[str]:
        """Return bound real names, excluding default entries.

        Used to detect whether a candidate memory names another speaker.
        """
        return [e.name for vpid, e in self._entries.items() if e.name and e.name != vpid]

    def to_dict(self) -> dict:
        return {
            vpid: {"role": e.role, "name": e.name, "entity_id": e.entity_id}
            for vpid, e in self._entries.items()
        }


# ── Emotion mapping (emotion labels -> VoiceMem affect vocabulary) ───────────
# This table originally matched emotion2vec's fixed nine-class vocabulary.
# paper_emotion_detector.py now produces free-form Qwen2.5-Omni labels, so exact
# matching often misses variants such as happiness, frustration, and anxiety.
# As a fallback, normalize_emotion() performs the same eight-class keyword
# matching used by the right-brain anchors before mapping to affect labels.
_EMO_TO_AFFECT: dict[str, str] = {
    "开心": "excited",  "快乐": "excited",  "happy": "excited",
    "悲伤": "sad",      "难过": "sad",       "sad": "sad",
    "愤怒": "angry",    "生气": "angry",     "angry": "angry",
    "焦虑": "anxious",  "恐惧": "anxious",   "fearful": "anxious",
    "厌恶": "disgusted","disgusted": "disgusted",
    "惊讶": "curious",  "surprised": "curious",
    "满意": "satisfied","satisfied": "satisfied",
}

# Fallback mapping from anchor_router._CANONICAL_EMOTIONS (8 classes) to affect.
_CANONICAL_TO_AFFECT: dict[str, str] = {
    "开心": "excited",
    "悲伤": "sad",
    "委屈": "angry",
    "孤独": "sad",
    "纠结": "anxious",
    "平静": "satisfied",
    "焦虑": "anxious",
    "疲惫": "sad",
}


def emotion_to_affect(emotion: str) -> str | None:
    """Map an emotion label to VoiceMem's affect field."""
    if not emotion:
        return None
    e = emotion.strip()
    direct = _EMO_TO_AFFECT.get(e)
    if direct:
        return direct
    from voicemem.rightbrain.anchor_router import normalize_emotion
    canonical = normalize_emotion(e)
    if canonical == "平静" and e not in ("平静", "calm", "neutral"):
        # normalize_emotion falls back to the neutral label for unknown input. Distinguish
        # a genuine calm label from an unknown value to avoid hiding gaps.
        return None
    return _CANONICAL_TO_AFFECT.get(canonical)


# ── Core adapter ─────────────────────────────────────────────────────────────

def _looks_like_voiceprint_id(vpid: str) -> bool:
    """Return whether this is a generated voiceprint ID or a supplied person name.

    Generated IDs look like person_86fed148 / utt_9f2a / spk_3. Treating a
    supplied name as an unknown identity would rewrite the memory subject as
    "User" and make name-based retrieval fail.
    """
    v = (vpid or "").lower()
    return v.startswith(("person_", "utt_", "spk_", "speaker_", "voiceprint_"))


def voice_input_to_messages(
    vi: VoiceInput,
    registry: VoiceprintRegistry,
) -> list[dict[str, str]]:
    """Convert VoiceInput.contents into OpenAI message format.

        - Merge consecutive sentences from the same voiceprint into one message.
        - Replace "Speaker N" with a registered real name when available.
        - Do not expose an unregistered person_id as a human name to the extractor.
            Use an explicit unidentified-speaker label while retaining the ID suffix,
            so different unknown speakers cannot be merged accidentally.
    """
    if not vi.contents:
        return []

    messages: list[dict[str, str]] = []
    current_vpid: str | None = None
    current_sentences: list[str] = []

    def _flush() -> None:
        if not current_sentences or current_vpid is None:
            return
        entry = registry.get(current_vpid)
        label = entry.name
        if label == current_vpid:
            if current_vpid.lower() in ("user", "voice_demo_user"):
                # Caller-supplied account-owner channel (text-mode demos pass
                # the literal id "user" for every utterance): this IS the
                # account owner by definition, so the defensive unidentified
                # label below would be actively wrong here -- it made every
                # stored fact read "Unidentified speaker user stated..."
                # (real user complaint). That label exists for unverified
                # VOICEPRINT ids, where assuming account-owner identity
                # mis-attributes speech across real people; a fixed text
                # channel has no such ambiguity. Once the user self-
                # identifies themselves, the registry binding (see
                # core.py's text-mode binding) replaces this with their
                # actual name via the normal entry.name path above.
                label = "User"
            elif _looks_like_voiceprint_id(current_vpid):
                # This is an unregistered voiceprint. The label must not be
                # stored as a person's name, and should remain a short neutral
                # marker so it does not change the language of the extraction.
                label = "Speaker 0"
            # Otherwise the caller supplied a person name directly. An absent
            # registry entry does not make that name unidentified, so preserve it.
        content = f"{label}: " + " ".join(current_sentences)
        messages.append({"role": entry.role, "content": content})

    for c in vi.contents:
        if not c.sentence:
            continue
        if c.voiceprint_id != current_vpid:
            _flush()
            current_vpid = c.voiceprint_id
            current_sentences = [c.sentence]
        else:
            current_sentences.append(c.sentence)

    _flush()

    # Append the agent side last: references such as "that one" are resolved
    # only with the assistant reply.
    if vi.agent_reply and vi.agent_reply.strip():
        messages.append({"role": "assistant", "content": vi.agent_reply.strip()})

    return messages


# ── Slot mapping ──────────────────────────────────────────────────────────────
# Targets are real SlotV2 enum values (work/finance/relationships/health/goals/
# daily_life/knowledge; see cognitive_graph/slot_v2.py). Older mappings used
# values outside the taxonomy and incorrect store attributes, causing silent
# failures in this path.

_VOICE_SLOT_TO_SLOTV2: dict[str, str] = {
    "work": "work", "task": "work", "todo": "work", "deadline": "work",
    "project": "work", "meeting": "work",
    "finance": "finance", "money": "finance", "salary": "finance",
    "goal": "goals", "plan": "goals",
    "fact": "knowledge", "knowledge": "knowledge", "info": "knowledge",
    "event": "daily_life", "experience": "daily_life", "memory": "daily_life",
    "preference": "daily_life", "like": "daily_life", "dislike": "daily_life",
    "routine": "daily_life", "habit": "daily_life", "daily": "daily_life",
    "relationship": "relationships", "people": "relationships",
    "family": "relationships", "friend": "relationships",
    "health": "health", "medical": "health",
    "place": "daily_life", "location": "daily_life",
}


def map_voice_slots_to_slotv2(voice_slots: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for s in voice_slots:
        mapped = _VOICE_SLOT_TO_SLOTV2.get(s.lower().strip())
        if mapped and mapped not in seen:
            seen.add(mapped)
            result.append(mapped)
    return result


# ── Ingest ───────────────────────────────────────────────────────────────────

@dataclass
class VoiceIngestResult:
    voice_id: str
    memory_ids: list[str]
    facts_count: int
    begin_time: str
    end_time: str
    slots: list[str]
    messages_count: int
    affect: str | None = None   # emotion2vec affect signal
    error: str | None = None


def ingest_voice_input(
    vi: VoiceInput,
    user_id: str,
    *,
    registry: VoiceprintRegistry,
    repo: Any,
    extractor: Any,
    extra_metadata: dict | None = None,
    session_id: int | str | None = None,
) -> VoiceIngestResult:
    """Inject one VoiceInput into the left-brain memory store.

        Pipeline: convert contents to messages, extract atomic facts, resolve
        conflicts, apply ADD/UPDATE/DELETE decisions, and pre-classify slots into
        memory_tags.
    """
    messages = voice_input_to_messages(vi, registry)
    if not messages:
        return VoiceIngestResult(
            voice_id=vi.id, memory_ids=[], facts_count=0,
            begin_time=vi.begin_time, end_time=vi.end_time,
            slots=vi.slots, messages_count=0, error="empty_contents",
        )

    # Use the current speaker's real name only when it is registered. Passing an
    # unregistered voiceprint ID as the speaker would mislead ConflictResolver.
    # This anchors which person said the new facts during conflict resolution.
    speaker_name: str | None = None
    if vi.contents:
        vpid = vi.contents[0].voiceprint_id
        resolved = registry.display_name(vpid)
        if resolved and resolved != vpid:
            speaker_name = resolved

    # Known other-person names. Candidates are searched by account user_id rather
    # than speaker, so another person's similar fact can enter the pool. The
    # cross-person UPDATE prohibition is only a soft prompt; filter candidates
    # that name another person unless they also name the current speaker.
    other_person_names = (
        [n for n in registry.all_display_names() if n != speaker_name]
        if speaker_name else []
    )

    def _drop_other_named_people(cands: list[dict[str, str]]) -> list[dict[str, str]]:
        if not speaker_name or not other_person_names:
            return cands
        out = []
        for c in cands:
            text = str(c.get("text", ""))
            if speaker_name in text:
                out.append(c)
                continue
            if any(name in text for name in other_person_names):
                continue
            out.append(c)
        return out

    # ── Step 1.5: candidate memories for extraction deduplication ─────────────
    # Previously the extractor received no existing_memories and could not see
    # The extractor's Existing Memories section is used for deduplication and
    # linked_memory_ids. Retrieve candidates from the whole turn first, but use
    # only the user side because assistant replies are longer and would crowd out
    # relevant memories from the top ten.
    query_text = "\n".join(str(m.get("content", "")) for m in messages
                           if m.get("role") != "assistant").strip()
    existing_for_extraction: list[dict[str, str]] = []
    if query_text and hasattr(repo, "search"):
        try:
            existing_for_extraction = [{"id": h.memory_id, "text": h.text}
                                        for h in repo.search(query_text, user_id=user_id, top_k=10)]
        except Exception:
            pass
    if not existing_for_extraction and hasattr(repo, "existing_for_extractor"):
        try:
            existing_for_extraction = repo.existing_for_extractor(user_id=user_id)
        except Exception:
            pass
    existing_for_extraction = _drop_other_named_people(existing_for_extraction)

    # ── Step 2: extract atomic facts ─────────────────────────────────────────
    # Optional raw fallback: if OpenAI extraction fails or returns no facts,
    # store the original sentence so the local E5 index can still retrieve it.
    def _raw_fallback() -> list:
        # Disabled by default. Set VOICEMEM_INGEST_RAW_FALLBACK=1 for offline or
        # no-key demos to retain the original text when extraction fails.
        if os.environ.get("VOICEMEM_INGEST_RAW_FALLBACK", "0") != "1":
            return []
        raw = " ".join(c.sentence for c in vi.contents if c.sentence).strip()
        if not raw:
            return []
        from voicemem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
        return [ExtractedAdditiveMemory(local_id="0", text=raw, attributed_to="user")]

    try:
        extracted = extractor.extract(
            new_messages=messages,
            existing_memories=existing_for_extraction,
            observation_date=vi.begin_time,
            current_date=vi.begin_time,
        )
    except Exception as e:
        extracted = _raw_fallback()
        fallback_status = "salvo il testo originale" if extracted else "nessun testo, salto"
        print(f"[ingest] estrazione fallita ({e}) -> {fallback_status}", flush=True)
        if not extracted:
            return VoiceIngestResult(
                voice_id=vi.id, memory_ids=[], facts_count=0,
                begin_time=vi.begin_time, end_time=vi.end_time,
                slots=vi.slots, messages_count=len(messages),
                error=f"extraction_failed: {e}",
            )

    if not extracted:
        extracted = _raw_fallback()
        if extracted:
            print("[ingest] nessun fatto estratto -> salvo il testo originale", flush=True)
        else:
            return VoiceIngestResult(
                voice_id=vi.id, memory_ids=[], facts_count=0,
                begin_time=vi.begin_time, end_time=vi.end_time,
                slots=vi.slots, messages_count=len(messages),
            )

    # ── Step 3-4: conflict resolution (Mem0 V1 style) ───────────────────────
    from voicemem.leftbrain.extract_facts_openai import ConflictResolver

    # Search each new fact independently, matching mem0's behavior. Searching the
    # whole turn once diluted the signal when several topics were discussed.
    # The wider window and attribute-only second query keep old values visible
    # for UPDATE/DELETE decisions. VOICEMEM_CONFLICT_WIDE=0 restores the old
    # top-5 behavior without the attribute query.
    wide = os.environ.get("VOICEMEM_CONFLICT_WIDE", "1") != "0"
    new_fact_texts = [m.text for m in extracted if m.text]
    existing_map: dict[str, dict[str, str]] = {}
    if hasattr(repo, "search"):
        queries: list[tuple[str, int]] = [(t, 15 if wide else 5) for t in new_fact_texts]
        if wide:
            queries += [(m.attribute, 10) for m in extracted
                        if m.text and getattr(m, "attribute", "")]
        for q, k in queries:
            try:
                for h in repo.search(q, user_id=user_id, top_k=k):
                    existing_map[h.memory_id] = {"id": h.memory_id, "text": h.text}
            except Exception:
                continue
    existing = list(existing_map.values())
    if not existing and hasattr(repo, "existing_for_extractor"):
        try:
            existing = repo.existing_for_extractor(user_id=user_id)
        except Exception:
            pass
    existing = _drop_other_named_people(existing)

    # Conflict resolution is one LLM call and becomes slower as the memory store
    # grows. VOICEMEM_ALWAYS_ADD=1 skips it and always appends; retrieval then
    # relies on timestamps to prefer the latest value.
    always_add = _os.environ.get("VOICEMEM_ALWAYS_ADD", "0") == "1"

    resolutions = []
    if existing and new_fact_texts and not always_add:
        try:
            resolver = ConflictResolver(single_valued_rule=wide)
            resolutions = resolver.resolve(new_fact_texts, existing, speaker_name=speaker_name)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("ConflictResolver failed, falling back to ADD-only: %s", e)

    # ── Step 5: apply decisions ───────────────────────────────────────────────
    # Apply resolver decisions when available; otherwise use the original ADD-only path.
    if resolutions:
        # Map fact text to ExtractedAdditiveMemory so ADD can reuse metadata.
        fact_map = {m.text: m for m in extracted if m.text}
        to_add: list = []
        for r in resolutions:
            if r.event == "ADD" and r.text:
                orig = fact_map.get(r.text) or next(iter(fact_map.values()), None)
                if orig:
                    from voicemem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
                    to_add.append(ExtractedAdditiveMemory(
                        local_id=r.memory_id,
                        text=r.text,
                        attributed_to=orig.attributed_to,
                    ))
            elif r.event == "UPDATE" and r.memory_id and r.text:
                # UPDATE is treated as ADD by default rather than rewriting old
                # memories. A wrong overwrite loses information permanently,
                # while a duplicate can be deduplicated later. Set
                # VOICEMEM_APPLY_UPDATE=1 to restore the legacy behavior.
                if os.environ.get("VOICEMEM_APPLY_UPDATE", "0") == "1" and hasattr(repo, "update_memory"):
                    # Carry the current session date so an updated fact does not
                    # retain the old observation timestamp.
                    repo.update_memory(r.memory_id, r.text, session_id=session_id,
                                       observed_at=vi.begin_time, user_id=user_id)
                else:
                    # Reuse ADD metadata from fact_map so attributed_to is not
                    # lost and the unverified-speaker safeguard remains intact.
                    orig = fact_map.get(r.text) or next(iter(fact_map.values()), None)
                    if orig:
                        from voicemem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
                        to_add.append(ExtractedAdditiveMemory(
                            local_id=r.memory_id, text=r.text,
                            attributed_to=orig.attributed_to,
                        ))
            elif r.event == "DELETE" and r.memory_id:
                if hasattr(repo, "delete_memory"):
                    repo.delete_memory(r.memory_id)
        extracted = to_add  # Keep only the ADD entries.

    # Collect entity bindings for all voiceprints in this batch.
    speaker_entity_map = {
        vpid: registry.entity_id(vpid)
        for vpid in {c.voiceprint_id for c in vi.contents}
        if registry.entity_id(vpid)
    }

    # emotion2vec → affect
    affect = emotion_to_affect(vi.dominant_emotion())

    meta = {
        "turn_id":            vi.id,
        "voice_id":           vi.id,
        "time_start":         vi.begin_time,
        "time_end":           vi.end_time,
        "voice_slots":        vi.slots,
        "source":             "voice",
        "speaker_entity_map": speaker_entity_map,
        "affect":             affect,
        **({"session_id": session_id} if session_id is not None else {}),
        **({"background_sounds": vi.environment} if vi.environment else {}),
        **(extra_metadata or {}),
    }
    memory_ids = repo.append_extracted(extracted, user_id=user_id, extra_metadata=meta)
    print(f"[ingest] salvate {len(memory_ids or [])} memorie: "
          f"{[m.text[:20] for m in extracted][:3]}", flush=True)

    # Write pre-classified slots to memory_tags.
    slotv2_hints = map_voice_slots_to_slotv2(vi.slots)
    if slotv2_hints and memory_ids:
        _write_slotv2_hints(repo, user_id, memory_ids, slotv2_hints)

    return VoiceIngestResult(
        voice_id=vi.id,
        memory_ids=memory_ids or [],
        facts_count=len(extracted),
        begin_time=vi.begin_time,
        end_time=vi.end_time,
        slots=vi.slots,
        messages_count=len(messages),
        affect=affect,
    )


def _write_slotv2_hints(repo: Any, user_id: str, memory_ids: list[str], slotv2_tags: list[str]) -> None:
    """Write coarse voice-module slot hints with lower confidence.

    They share memory_tags with LLM and embedding-based labels, so they must not
    overwrite a more reliable classification.
    """
    try:
        store = repo._cognitive_store  # type: ignore[attr-defined]
        if store is None or not hasattr(store, "upsert_memory_tags"):
            return
        tags = [(slot, 0.5) for slot in slotv2_tags]
        for mid in memory_ids:
            store.upsert_memory_tags(mid, user_id, tags)
    except Exception:
        pass
