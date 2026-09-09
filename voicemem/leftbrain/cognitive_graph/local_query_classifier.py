"""Classificatore query locale: query → slots(+entities), senza toccare LLM/rete."""

# Allineato all'interfaccia di ``QuerySlotClassifier`` integrato (1 chiamata LLM), può essere sostituito tramite ``VoiceMem(schema=...)``
# iniettandolo, spostando il passo ``Classify`` (estrai slot + entity) dall'API OpenAI al modello locale —
# completamente simmetrico all'iniezione di ``embedding``:

    from voicemem import VoiceMem
    from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
    vm = VoiceMem(schema=lambda: LocalQueryClassifier())   # slots 走本地 E5，0 LLM
    vm.search("Dove lavoro?")                                  # Classify non chiama più LLM

# Scelte progettuali (spiegate onestamente, nascosto nulla):
# - **slots**: cosine E5 vs top-k delle descrizioni dei 7 slot base (~93% coerente con LLM nei test).
# - **entity**: default vuoto → usa il restringimento slot-only esistente di voicemem (non uno stato nuovo negativo). Il riconoscimento
#   di entità open-source **non richiede obbligatoriamente LLM**: passa ``ner=<callable: query -> list[str]>`` per integrare NER locale
#   (gliner / spaCy ecc.) e localizzare anche le entity.
# - **non implementa classify_child**: il "drill-down" degli slot emergenti viene affidato alla versione LLM; la nascita di slot secondari avviene
#   comunque nel lato scrittura, non nel percorso caldo della ricerca. ``engine.Classify`` rileva automaticamente l'assenza di ``classify_child``
#   e salta il drill-down, usando solo base-7.

I prefissi ``"query: "`` / ``"passage: "`` di E5 sono obbligatori (non decorativi). Il modello viene scaricato automaticamente al primo utilizzo;
passa ``model=`` per riutilizzare un SentenceTransformer già caricato (es. condiviso con l'embedder locale, risparmia memoria).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification

# Condivide la stessa istanza E5 dei vettori memoria (models/embedding/), risparmia un set di pesi
def _model_name() -> str:
    from voicemem.utils.common.paths import hf_model
    return hf_model("embedding", "intfloat/multilingual-e5-small", "VOICEMEM_E5_MODEL")


_MODEL_NAME = _model_name()

# Stesse descrizioni dei 7 slot base del classificatore LLM integrato (volontariamente coerenti, approssimazione locale dello stesso processo decisionale di classificazione).
_SLOT_DESCRIPTIONS = {
    "work": "career, job, company, projects, colleagues, workplace, 工作, 职业, 辞职, 升职",
    "finance": "money, salary, income, expenses, investments, savings, 财务, 薪资, 投资",
    "relationships": "friends, family, romantic, social connections, 朋友, 家人, 感情",
    "health": "physical health, exercise, diet, sleep, medical, 健康, 运动, 生病",
    "goals": "future plans, dreams, aspirations, self-improvement, 目标, 计划, 梦想",
    "daily_life": "daily routines, hobbies, leisure, lifestyle, 日常, 爱好, 习惯",
    "knowledge": "learning, concepts, skills, facts, technology, 知识, 技能, 学习",
}


class LocalQueryClassifier:
    """query → QueryClassification(slots, entities)，本地 E5，无 LLM/网络。"""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        model=None,
        ner: Callable[[str], Sequence[str]] | None = None,
        top_k: int = 2,
    ) -> None:
        self._model_name = model_name
        self._model = model                 # 可复用已加载的 SentenceTransformer
        self._ner = ner                     # 可选本地实体识别（默认无 → slot-only）
        self._top_k = top_k
        self._slot_names = list(_SLOT_DESCRIPTIONS)
        self._slot_embs = None              # 懒算一次的槽描述向量

    def _m(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _slots_matrix(self):
        if self._slot_embs is None:
            texts = [f"passage: {k}: {v}" for k, v in _SLOT_DESCRIPTIONS.items()]
            self._slot_embs = np.asarray(self._m().encode(texts, normalize_embeddings=True))
        return self._slot_embs

    def classify(self, query: str, extra_slots=None) -> QueryClassification:
        """base-7 里挑 top-k 个 slot（E5 余弦）+ 可选本地实体。extra_slots（动态子
        slot 候选）本地版忽略——子 slot 下钻交给 LLM 版/写入侧维护。"""
        q = np.asarray(self._m().encode([f"query: {query}"], normalize_embeddings=True)[0])
        order = np.argsort(-(self._slots_matrix() @ q))[: self._top_k]
        slots = [self._slot_names[i] for i in order]
        entities = list(self._ner(query)) if self._ner else []
        return QueryClassification(slots=slots, entities=entities)
