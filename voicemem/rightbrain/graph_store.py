"""Layer slot→entity→memory del cervello destro.

Stessa struttura a tre livelli del cervello sinistro, ma il contenuto è sensibile/soggettivo, non la classificazione fattuale del cervello sinistro:

  slot (5 categorie iniziali: emozione / preferenze e avversioni / stile espressivo / modello mentale / modalità di coping)
    └── entity (nodi sensitivi specifici, es. "felice" sotto "emozione", "essere interrotto" sotto "preferenze e avversioni")
          └── memory (id di memoria specifici appesi a un'entity, puntano a right_brain_memories.id)

Lo slot è qui il vero nodo del grafo (con il proprio id + description), non un attributo stringa come nel cervello sinistro.

La deduplicazione delle entity si basa sulla similarità semantica (non corrispondenza esatta di stringhe): l'etichetta sensitiva appena estratta viene prima confrontata in similarità con l'embedding delle entity già esistenti nello stesso slot, se simile abbastanza viene riutilizzata, altrimenti ne viene creata una nuova.
"""

from __future__ import annotations

import os

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from voicemem.utils.common._graph_common import cosine as _cosine, new_id as _new_id, utc_iso as _utc_iso

#: A che punto un nuovo tag è simile abbastanza alle entity esistenti nello stesso slot da essere considerato lo stesso.
#:
#: Originariamente 0.65 — per le frasi cinesi di E5 equivale a "tutto fuso": la baseline è già alta nei test reali,
#: "odia quando mangia rumorosamente" e "odia essere interrotto" hanno 0.934, "odia le riunioni lunghe" ha 0.922, tutti
#: raggruppati in una singola entity. Il risultato è che dopo un po' il cervello destro smette di creare nuovi nodi, qualsiasi cosa nuova dica l'utente non ha reazione sulla mappa,
#: sembra come se non ricordasse nulla.
#: I casi che dovrebbero davvero essere fusi ("ama il caffè fatto a mano" ↔ "preferisce il caffè fatto a mano") sono 0.964,
#: 0.95 separa esattamente le due categorie.
DEFAULT_MATCH_THRESHOLD = float(os.environ.get("VOICEMEM_RB_ENTITY_MERGE", "0.95"))


#: I 5 slot sensitivi iniziali; description lasciato vuoto per ora, da completare dopo.
# Le entity iniziali di emotion riutilizzano le 8 etichette emotive esistenti.
SEED_SLOTS: list[tuple[str, str, list[str]]] = [
    ("Emozione", "", ["ansia", "tristezza", "ingiustizia", "solitudine", "confusione", "pace", "gioia", "stanchezza"]),
    ("Preferenze e avversioni", "", []),
    ("Stile espressivo", "", []),
    ("Modello mentale", "", []),
    ("Modalità di coping", "", []),
]


@dataclass
class RBSlot:
    id: str
    user_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_utc_iso)


@dataclass
class RBEntity:
    id: str
    user_id: str
    slot_id: str
    name: str
    description: str = ""
    embedding: list[float] | None = None
    source_entity_id: str | None = None  # Entity.id del cognitive_graph del cervello sinistro, compilato solo per entity di tipo "nodo relazione"
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)


class RightBrainGraphStore:
    """SQLite per memorizzare/recuperare il grafo a tre livelli slot→entity→memory del cervello destro. Thread-safe: crea una nuova connessione per ogni operazione."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS rb_slots (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                UNIQUE (user_id, name)
            );

            CREATE TABLE IF NOT EXISTS rb_entities (
                id                TEXT PRIMARY KEY,
                user_id           TEXT NOT NULL,
                slot_id           TEXT NOT NULL,
                name              TEXT NOT NULL,
                description       TEXT NOT NULL DEFAULT '',
                embedding         TEXT,
                source_entity_id  TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                UNIQUE (user_id, slot_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_rbe_slot ON rb_entities(user_id, slot_id);

            CREATE TABLE IF NOT EXISTS rb_entity_memories (
                id          TEXT PRIMARY KEY,
                entity_id   TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                memory_id   TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE (entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_rbem_entity ON rb_entity_memories(entity_id);
            CREATE INDEX IF NOT EXISTS idx_rbem_memory ON rb_entity_memories(user_id, memory_id);
            """)
            # Migrazione: il vecchio database (CREATE TABLE IF NOT EXISTS non funziona su tabelle esistenti) aggiunge
            # la colonna source_entity_id — tutti i rb_graph.sqlite creati prima del lancio della funzione nodo relazione mancano di questa colonna.
            cols = {row["name"] for row in c.execute("PRAGMA table_info(rb_entities)")}
            if "source_entity_id" not in cols:
                c.execute("ALTER TABLE rb_entities ADD COLUMN source_entity_id TEXT")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_rbe_source "
                "ON rb_entities(user_id, slot_id, source_entity_id)"
            )

    # ── Inizializzazione seed ─────────────────────────────────────────────────────────────

    def ensure_seed_slots(self, user_id: str) -> None:
        """Assicura che questo utente abbia già 5 slot iniziali (+ 8 entity iniziali sotto Emozione). Idempotente, può essere chiamato ripetutamente."""
        for slot_name, slot_desc, seed_entities in SEED_SLOTS:
            slot = self.get_or_create_slot(user_id, slot_name, description=slot_desc)
            for ent_name in seed_entities:
                self.get_or_create_entity(user_id, slot.id, ent_name)

    # ── Slot ──────────────────────────────────────────────────────────────────

    def get_or_create_slot(self, user_id: str, name: str, *, description: str = "") -> RBSlot:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? AND name=?", (user_id, name)
            ).fetchone()
            if row:
                return _row_to_slot(row)
            slot = RBSlot(id=_new_id(), user_id=user_id, name=name, description=description)
            c.execute(
                "INSERT INTO rb_slots (id, user_id, name, description, created_at) VALUES (?,?,?,?,?)",
                (slot.id, slot.user_id, slot.name, slot.description, slot.created_at),
            )
            return slot

    def get_slot_by_name(self, user_id: str, name: str) -> RBSlot | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? AND name=?", (user_id, name)
            ).fetchone()
        return _row_to_slot(row) if row else None

    def list_slots(self, user_id: str) -> list[RBSlot]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? ORDER BY name", (user_id,)
            ).fetchall()
        return [_row_to_slot(r) for r in rows]

    def set_slot_description(self, slot_id: str, description: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE rb_slots SET description=? WHERE id=?", (description, slot_id))

    # ── Entity (corrispondenza nome esatta, adatta per seed/glossari fissi) ────────────────────────

    def get_or_create_entity(
        self,
        user_id: str,
        slot_id: str,
        name: str,
        *,
        description: str = "",
        embedding: list[float] | None = None,
    ) -> RBEntity:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
            if row:
                return _row_to_entity(row)
            return self._insert_entity(c, user_id, slot_id, name, description, embedding)

    # ── Entity (corrispondenza similarità semantica, adatta per entity sensitive generate liberamente) ───────────────────

    def find_similar_entity(
        self,
        user_id: str,
        slot_id: str,
        embedding: list[float],
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> RBEntity | None:
        """Trova l'entity semanticamente più simile nello stesso slot, restituisce None se la similarità non è sufficiente."""
        best: RBEntity | None = None
        best_sim = -1.0
        for ent in self.get_entities_for_slot(user_id, slot_id):
            if ent.embedding is None:
                continue
            sim = _cosine(embedding, ent.embedding)
            if sim > best_sim:
                best_sim, best = sim, ent
        return best if best is not None and best_sim >= threshold else None

    def get_or_create_entity_semantic(
        self,
        user_id: str,
        slot_id: str,
        name: str,
        embedding: list[float],
        *,
        description: str = "",
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> tuple[RBEntity, bool]:
        """Prima cerca un'entity esistente per similarità semantica, se trovata la riutilizza; altrimenti ne crea una nuova.

        Returns
        -------
        (entity, created) — created=True significa che è stata creata una nuova entity questa volta.
        """
        existing = self.find_similar_entity(user_id, slot_id, embedding, threshold=threshold)
        if existing is not None:
            return existing, False
        with self._conn() as c:
            ent = self._insert_entity(c, user_id, slot_id, name, description, embedding)
        return ent, True

    def _insert_entity(
        self,
        c: sqlite3.Connection,
        user_id: str,
        slot_id: str,
        name: str,
        description: str,
        embedding: list[float] | None,
        source_entity_id: str | None = None,
    ) -> RBEntity:
        now = _utc_iso()
        ent = RBEntity(
            id=_new_id(), user_id=user_id, slot_id=slot_id, name=name,
            description=description, embedding=embedding,
            source_entity_id=source_entity_id, created_at=now, updated_at=now,
        )
        c.execute(
            """INSERT INTO rb_entities
               (id, user_id, slot_id, name, description, embedding, source_entity_id,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ent.id, ent.user_id, ent.slot_id, ent.name, ent.description,
             json.dumps(embedding) if embedding is not None else None,
             ent.source_entity_id, ent.created_at, ent.updated_at),
        )
        return ent

    # ── Entity (corrispondenza precisa per entity.id del cervello sinistro, adatta per scenari "nodo relazione") ────────────────
    # Un'entity del cervello sinistro (persona/luogo/progetto...) corrisponde sempre allo stesso nodo relazione del cervello destro, tramite source_entity_id
    # corrispondenza esatta, non similarità semantica — la similarità semantica è per deduplicare etichette astratte generate liberamente, qui
    # non serve: il renaming/fusione delle entity del cervello sinistro non influisce sulla corrispondenza qui (l'ID non cambia), stessa logica di oggi di
    # usare entity.id reali negli anchor invece di stringhe di nome.

    def get_entity_by_source_id(
        self, user_id: str, slot_id: str, source_entity_id: str,
    ) -> RBEntity | None:
        with self._conn() as c:
            row = c.execute(
                """SELECT * FROM rb_entities
                   WHERE user_id=? AND slot_id=? AND source_entity_id=?""",
                (user_id, slot_id, source_entity_id),
            ).fetchone()
        return _row_to_entity(row) if row else None

    def get_or_create_entity_by_source_id(
        self, user_id: str, slot_id: str, source_entity_id: str, name: str,
    ) -> tuple[RBEntity, bool]:
        """Trova/crea nodo relazione per entity.id del cervello sinistro.

        Returns
        -------
        (entity, created) — created=True significa che è stata creata una nuova entity questa volta.
        """
        existing = self.get_entity_by_source_id(user_id, slot_id, source_entity_id)
        if existing is not None:
            return existing, False
        with self._conn() as c:
            # Nodo con lo stesso nome esiste già ma source_entity_id non corrisponde: il cervello sinistro può dare
            # più di un entity.id per la stessa cosa reale (entità omonime vengono re-estratte nelle utterance successive/
            # cambiano entity_type), e rb_entities ha UNIQUE(user_id,
            # slot_id, name) — INSERT diretto causerebbe violazione vincolo unico con eccezione, il chiamante
            # (loop nodo relazione in core.py::_finish_ingest) interromperebbe l'intero blocco, perdendo
            # tutti gli anchor e link_memory delle entity rimanenti in questa
            # utterance. Stesso nome = stessa entità,
            # riutilizza il nodo esistente; se il nodo esistente non ha ancora claimato source_entity_id, fallo ora,
            # così le ricerche future possono usare il percorso più veloce source_id.
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
            if row is not None:
                ent = _row_to_entity(row)
                if not ent.source_entity_id:
                    now = _utc_iso()
                    c.execute(
                        "UPDATE rb_entities SET source_entity_id=?, updated_at=? WHERE id=?",
                        (source_entity_id, now, ent.id),
                    )
                    ent.source_entity_id = source_entity_id
                    ent.updated_at = now
                return ent, False
            ent = self._insert_entity(
                c, user_id, slot_id, name, "", None, source_entity_id=source_entity_id,
            )
        return ent, True

    def get_entities_for_slot(self, user_id: str, slot_id: str,
                              *, newest_first: bool = False) -> list[RBEntity]:
        """Ordina per nome di default (stabile, facile da visualizzare). ``newest_first`` ordina inversamente per ordine di inserimento —
        la mappa cerebrale deve riservare alcuni posti alle entità appena cresciute, ordinando per nome dove finiscono le nuove dipende solo da come si chiamano."""
        order = "rowid DESC" if newest_first else "name"
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? ORDER BY {order}",
                (user_id, slot_id),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def get_entity_by_name(self, user_id: str, slot_id: str, name: str) -> RBEntity | None:
        """Cerca precisa per nome (sola lettura, non crea) — per il retrieval, a differenza di get_or_create_entity
        che ne crea uno vuoto quando non trova nulla."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
        return _row_to_entity(row) if row else None

    def get_entity(self, entity_id: str) -> RBEntity | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM rb_entities WHERE id=?", (entity_id,)).fetchone()
        return _row_to_entity(row) if row else None

    def get_slot(self, slot_id: str) -> RBSlot | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM rb_slots WHERE id=?", (slot_id,)).fetchone()
        return _row_to_slot(row) if row else None

    def set_entity_description(self, entity_id: str, description: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE rb_entities SET description=?, updated_at=? WHERE id=?",
                (description, _utc_iso(), entity_id),
            )

    # ── Entity ↔ Memory ─────────────────────────────────────────────────────

    def link_memory(self, entity_id: str, user_id: str, memory_id: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO rb_entity_memories
                   (id, entity_id, user_id, memory_id, created_at)
                   VALUES (?,?,?,?,?)""",
                (_new_id(), entity_id, user_id, memory_id, _utc_iso()),
            )

    def get_memories_for_entity(self, entity_id: str) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT memory_id FROM rb_entity_memories WHERE entity_id=?", (entity_id,)
            ).fetchall()
        return [r["memory_id"] for r in rows]

    def get_entities_for_memory(self, user_id: str, memory_id: str) -> list[RBEntity]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT rbe.* FROM rb_entities rbe
                   JOIN rb_entity_memories rbem ON rbem.entity_id = rbe.id
                   WHERE rbem.user_id=? AND rbem.memory_id=?""",
                (user_id, memory_id),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]


def _row_to_slot(row: sqlite3.Row) -> RBSlot:
    return RBSlot(
        id=row["id"], user_id=row["user_id"], name=row["name"],
        description=row["description"], created_at=row["created_at"],
    )


def _row_to_entity(row: sqlite3.Row) -> RBEntity:
    return RBEntity(
        id=row["id"], user_id=row["user_id"], slot_id=row["slot_id"],
        name=row["name"], description=row["description"],
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        source_entity_id=row["source_entity_id"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )
