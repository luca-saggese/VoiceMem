"""Dopo aver cambiato embedder, ricalcola i vettori obsoleti per dimensione nel database.

Quando serve: `_embed_text` segue ora l'embedder iniettato, mentre
`rb_traits` / `graph_entities` potrebbero contenere vettori calcolati dal vecchio embedder.
Quelli con dimensione non corrispondente vengono saltati (con warning), il retrieval del cervello destro e la deduplicazione delle entity falliscono finché non si re-embed.

Esegui: python3 tools/reembed.py <space> [--apply] [--local]
Senza --apply fa solo statistiche; --local usa la configurazione del web demo (E5 locale) per calcolare,
altrimenti usa l'embedder default (OpenAI). **Deve essere coerente con la tua configurazione di runtime effettiva**,
altrimenti la dimensione calcolata è sbagliata e non risolvi nulla.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    space = sys.argv[1] if len(sys.argv) > 1 else "demo"
    apply = "--apply" in sys.argv
    db = Path("voicemem_memoryspace") / space / f"{space}.sqlite"
    if not db.is_file():
        print(f"Non trovato {db}")
        return

    # Crea solo l'embedder, non l'intero VoiceMem — quest'ultimo aprirebbe il database vettoriale, contendendo il
    # file lock di qdrant ("Storage folder ... already accessed by another instance").
    # La migrazione modifica la colonna dei vettori nello sqlite, non ha nulla a che fare con il database vettoriale.
    if "--local" in sys.argv:                     # Allineato alla configurazione del web demo
        from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
        e = LocalE5Embedder()
        embed = e.embed_query_text
    else:
        from voicemem.leftbrain.local_memory_store import (
            OpenAILocalEmbedder, OpenAILocalEmbedderConfig,
        )
        e = OpenAILocalEmbedder(OpenAILocalEmbedderConfig())
        embed = lambda t: e.embed_texts([t])[0]
    want = len(embed("dimensione_probe"))
    print(f"{space}: embedder corrente produce {want} dimensioni")

    import json
    import numpy as np

    def vec_len(b):
        """I due tabelle hanno formati diversi: rb_traits è float32 binario, graph_entities è JSON."""
        if isinstance(b, (bytes, bytearray)):
            return len(np.frombuffer(b, dtype=np.float32))
        try:
            return len(json.loads(b))
        except Exception:
            return -1

    def pack(table, vec):
        return (np.asarray(vec, dtype=np.float32).tobytes() if table == "rb_traits"
                else json.dumps([float(x) for x in vec]))

    jobs = []          # (tabella, colonna id, righe da ricalcolare)
    con = sqlite3.connect(db)
    for table, idc, txtc in (("rb_traits", "id", "claim"),
                             ("graph_entities", "id", "name")):
        try:
            rows = con.execute(
                f"SELECT {idc}, {txtc}, embedding FROM {table} "
                "WHERE embedding IS NOT NULL").fetchall()
        except sqlite3.OperationalError:
            continue
        stale = [(i, t) for i, t, b in rows if vec_len(b) != want]
        print(f"  {table:16} totale {len(rows):4} righe, dimensione obsoleta {len(stale)}")
        if stale:
            jobs.append((table, idc, stale))

    if not jobs:
        print("Nessuna riga da ricalcolare.")
        return
    if not apply:
        print("(senza --apply, solo statistiche)")
        return

    total = 0
    for table, idc, stale in jobs:
        for n, (mid, text) in enumerate(stale, 1):
            con.execute(f"UPDATE {table} SET embedding=? WHERE {idc}=?",
                        (pack(table, embed(text)), mid))
            total += 1
            if n % 25 == 0:
                con.commit()
                print(f"  {table} …{n}/{len(stale)}")
        con.commit()
    print(f"\nRicalcolo completato: {total} righe")


if __name__ == "__main__":
    main()
