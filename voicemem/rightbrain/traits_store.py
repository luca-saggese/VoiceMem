"""Cervello destro v2: un nodo = un giudizio su questa persona, con evidenze appese sotto.

    rb_traits                          Nodi
      claim      Quando è sotto stress vuole essere rassicurato  Fissato alla scrittura, 5-15 caratteri
      slot       Uno su cinque (emozione/modalità di coping/stile espressivo/modello mentale/preferenze e avversioni)
      embedding  Vettore del claim — il cervello destro finalmente può fare retrieval semantico
    rb_evidence                        Evidenze
      quote      Non darmi soluzioni ora, lascia che finisca   Parola originale dell'utente
      emotion    Irritato                  L'emozione è un attributo dell'evidenza, non del nodo
      cause_id   ← quella fact del cervello sinistro

**Perché sostituire il vecchio slot → entity → heartnote**: il livello ``entity`` svolgeva tre ruoli —
a volte era un giudizio sulla persona ("odia essere interrotto"), a volte un argomento ("caffè fatto a mano", "NUS"), a volte una parola emotiva
("ansia"). Tre cose diverse mescolate in un livello, i risultati osservati nei test erano:

  · Tutte le cose tristi raggruppate in un singolo nodo "tristezza" (61 voci), tutte le conversazioni collegate a "Jiaqi" (52 voci)
  · Formato del titolo non uniforme — tre tipi di cose non hanno una scrittura unificata
  · Le descrizioni devono essere integrate con consolidamento batch post-hoc, eseguito raramente, quindi molti nodi senza descrizione

Qui vengono messi solo **giudizi sulla persona**. Le entità tematiche tornano al cognitive graph del cervello sinistro; le emozioni sono degradate ad attributi delle evidenze.
Le auto-riflessioni dell'assistant (response_experience) non entrano in questa tabella.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Due claim sono simili abbastanza da essere considerati lo stesso a questo livello, le evidenze vengono fuse dentro.
#: 0.95 è la soglia osservata nei test: la similarità baseline di E5 locale per frasi cinesi è già 0.9+,
#: "ama il caffè fatto a mano" ↔ "preferisce il caffè fatto a mano" è 0.964 (dovrebbe essere fuso),
#: "odia quando mangia rumorosamente" ↔ "odia essere interrotto" è 0.934 (non dovrebbe essere fuso).
MERGE_THRESHOLD = 0.95

#: Cinque slot. Rimossi i vecchi "persona/luogo/atteggiamento" — quello memorizzava argomenti (caffè fatto a mano/NUS/Jiaqi),
#: che dovrebbero appartenere al cervello sinistro, ed era proprio la fonte del miscuglio "Jiaqi ×52".
SLOTS = ("Emozione", "Modalità di coping", "Stile espressivo", "Modello mentale", "Preferenze e avversioni")

#: Cinque slot → tre cluster UI
SLOT_TO_CLUSTER = {
    "Emozione":       "emotion",
    "Modalità di coping":    "personality",
    "Stile espressivo":    "personality",
    "Modello mentale":    "personality",
    "Preferenze e avversioni":  "preference",
}


@dataclass
class Evidence:
    quote: str
    emotion: str = ""
    cause: str = ""            # Testo originale della fact del cervello sinistro (usato come "perché" durante il rendering)
    cause_id: str = ""
    at: str = ""


@dataclass
class Trait:
    id: str
    slot: str
    claim: str
    confidence: float = 0.9
    evidence: list[Evidence] = field(default_factory=list)
    updated_at: str = ""

    @property
    def cluster(self) -> str:
        return SLOT_TO_CLUSTER.get(self.slot, "personality")


#: Soggetti comuni prima di un claim. Il titolo del nodo è "odia essere interrotto" invece di "l'utente odia essere interrotto" —
#: l'intero grafo parla della stessa persona, avere "utente" davanti a ogni titolo è solo rumore.
_SUBJECTS = ("L'utente potrebbe", "L'utente sembra", "L'utente tende a", "L'utente", "Lui/lei", "L'altra parte", "Io")


def normalize_claim(claim: str) -> str:
    """Riformatta il claim in ciò che dovrebbe essere un titolo di nodo: senza soggetto, senza punto finale, una breve frase.

    La qualità del prodotto da diversi percorsi di scrittura varia — quello dell'estrazione combinata ha requisiti di formato espliciti, quello della riflessione dell'assistant
    (user_trait di response_experience) no, nei test sono stati prodotti output come
    "L'utente ama condividere le proprie esperienze, probabilmente non presta attenzione ai saluti dell'assistant." con soggetto intero.
    Invece di scrivere i requisiti su ogni percorso separatamente, è meglio uniformare all'ingresso.
    """
    c = (claim or "").strip().strip("「」\"'").rstrip("。.！!；;，,")
    for s in _SUBJECTS:
        if c.startswith(s) and len(c) > len(s) + 2:
            c = c[len(s):].lstrip("，,、 ")
            break
    # Frasi doppie tipo "A, forse B" — mantieni solo la prima metà — la seconda è quasi sempre un'ipotesi aggiunta dal modello
    if "，" in c and len(c) > 15:
        head = c.split("，")[0].strip()
        if len(head) >= 5:
            c = head
    return c.strip()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TraitStore:
    """Tabelle rb_traits / rb_evidence, condividono lo stesso sqlite di space con gli altri archivi strutturati."""

    def __init__(self, db_path, embed) -> None:
        self._db = str(db_path)
        self._embed = embed                 # fn(text) -> list[float]
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS rb_traits (
                id             TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                slot           TEXT NOT NULL,
                claim          TEXT NOT NULL,
                embedding      BLOB,
                confidence     REAL NOT NULL DEFAULT 0.9,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS rb_evidence (
                id         TEXT PRIMARY KEY,
                trait_id   TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                quote      TEXT NOT NULL,
                emotion    TEXT NOT NULL DEFAULT '',
                cause      TEXT NOT NULL DEFAULT '',
                cause_id   TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_ev_trait ON rb_evidence(trait_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tr_user ON rb_traits(user_id, slot)")

    def _conn(self):
        c = sqlite3.connect(self._db, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ── Scrittura ────────────────────────────────────────────────────────────────────

    def add(self, user_id: str, slot: str, claim: str, ev: Evidence) -> str:
        """Aggiunge un giudizio + le sue evidenze. Se esiste già un claim con significato identico viene fuso dentro, non crea un nuovo nodo."""
        claim = normalize_claim(claim)
        if not claim or slot not in SLOTS:
            return ""

        vec = self._vec(claim)
        tid = self._find_similar(user_id, slot, vec)
        now = _now()
        with self._conn() as c:
            if tid is None:
                tid = uuid.uuid4().hex
                c.execute("INSERT INTO rb_traits "
                          "(id,user_id,slot,claim,embedding,confidence,created_at,updated_at) "
                          "VALUES (?,?,?,?,?,?,?,?)",
                          (tid, user_id, slot, claim,
                           vec.astype(np.float32).tobytes() if vec is not None else None,
                           0.9, now, now))
            else:
                c.execute("UPDATE rb_traits SET updated_at=? WHERE id=?", (now, tid))
            # Ogni campo passa attraverso str(): le evidenze spesso provengono da vecchi dati o output LLM,
            # quando manca un campo è None, e queste colonne sono tutte NOT NULL, l'inserimento diretto causerebbe il fallimento dell'intera scrittura.
            c.execute("INSERT INTO rb_evidence "
                      "(id,trait_id,user_id,quote,emotion,cause,cause_id,created_at) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (uuid.uuid4().hex, tid, user_id, str(ev.quote or ""),
                       str(ev.emotion or ""), str(ev.cause or ""),
                       str(ev.cause_id or ""), str(ev.at or now)))
        return tid

    def _vec(self, text: str):
        try:
            v = np.asarray(self._embed(text), dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n else v
        except Exception:
            return None

    def _find_similar(self, user_id: str, slot: str, vec) -> str | None:
        if vec is None:
            return None
        with self._conn() as c:
            rows = c.execute("SELECT id, embedding FROM rb_traits "
                             "WHERE user_id=? AND slot=? AND embedding IS NOT NULL",
                             (user_id, slot)).fetchall()
        best, best_sim = None, 0.0
        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape != vec.shape:
                continue
            sim = float(v @ vec)
            if sim > best_sim:
                best, best_sim = r["id"], sim
        return best if best_sim >= MERGE_THRESHOLD else None

    # ── Lettura ────────────────────────────────────────────────────────────────────

    def all(self, user_id: str, *, per_slot: int = 8) -> list[Trait]:
        """Per la mappa cerebrale: da ogni slot prende le prime voci con più evidenze + quelle aggiunte più recentemente."""
        out: list[Trait] = []
        with self._conn() as c:
            for slot in SLOTS:
                rows = c.execute(
                    """SELECT t.*, COUNT(e.id) n FROM rb_traits t
                       LEFT JOIN rb_evidence e ON e.trait_id = t.id
                       WHERE t.user_id=? AND t.slot=? GROUP BY t.id
                       HAVING n > 0""", (user_id, slot)).fetchall()
                by_ev = sorted(rows, key=lambda r: -r["n"])
                by_new = sorted(rows, key=lambda r: r["updated_at"], reverse=True)
                fresh = max(1, per_slot // 2)
                picked, seen = [], set()
                # Metà per quelli aggiunti più recentemente (l'ultima frase detta deve essere visibile immediatamente), metà per quelli con più evidenze
                for r in by_new[:fresh] + by_ev:
                    if r["id"] in seen:
                        continue
                    seen.add(r["id"])
                    picked.append(r)
                    if len(picked) >= per_slot:
                        break
                for r in picked:
                    out.append(self._to_trait(c, r))
        return out

    def search(self, user_id: str, query: str, *, top_k: int = 5) -> list[Trait]:
        """Cerca giudizi per similarità semantica.

        Il vecchio cervello destro poteva solo corrispondere per anchor emotivo, quindi ogni turno restituiva sempre le stesse poche statiche descrizioni.
        Ora che il claim ha un vettore, qui è vero retrieval."
        """
        return [t for t, _ in self.search_scored(user_id, query, top_k=top_k)]

    def search_scored(self, user_id: str, query: str, *, top_k: int = 5
                      ) -> list[tuple[Trait, float]]:
        """Come :meth:`search`, ma include la similarità coseno.

        Il lato retrieval lo usa come priority — quanto un giudizio è rilevante per questa frase determina direttamente se dovrebbe occupare una posizione
        top-N; una priority fissa farebbe sì che giudizi irrilevanti spingano via quelli veramente rilevanti."
        """
        q = self._vec(query)
        if q is None:
            return []
        with self._conn() as c:
            rows = c.execute("SELECT * FROM rb_traits WHERE user_id=? AND embedding IS NOT NULL",
                             (user_id,)).fetchall()
            scored = []
            for r in rows:
                v = np.frombuffer(r["embedding"], dtype=np.float32)
                if v.shape != q.shape:
                    continue
                scored.append((float(v @ q), r))
            scored.sort(key=lambda t: -t[0])
            return [(self._to_trait(c, r), s) for s, r in scored[:top_k]]

    def _to_trait(self, c, r) -> Trait:
        evs = c.execute("SELECT * FROM rb_evidence WHERE trait_id=? ORDER BY created_at DESC",
                        (r["id"],)).fetchall()
        return Trait(
            id=r["id"], slot=r["slot"], claim=r["claim"],
            confidence=r["confidence"], updated_at=r["updated_at"],
            evidence=[Evidence(quote=e["quote"], emotion=e["emotion"],
                               cause=e["cause"], cause_id=e["cause_id"],
                               at=e["created_at"]) for e in evs],
        )

    def counts(self, user_id: str) -> tuple[int, int]:
        with self._conn() as c:
            t = c.execute("SELECT COUNT(*) FROM rb_traits WHERE user_id=?", (user_id,)).fetchone()[0]
            e = c.execute("SELECT COUNT(*) FROM rb_evidence WHERE user_id=?", (user_id,)).fetchone()[0]
        return t, e
