#!/usr/bin/env python3
"""Rivaluta un test già completato — senza ripetare retrieval e generazione.

    python evaluation/score.py --file results/locomo.json
    python evaluation/score.py --file results/locomo.json --judge gpt-4o --out results/locomo-gpt4o.json

Il retrieval e la generazione sono la metà costosa (una search + una generation per domanda), mentre il scoring è la metà economica. Separandoli:
cambiare modello judge, correggere bug nel criterio di valutazione, verificare se cambiando judge i punteggi sono stabili — tutto questo richiede di ripetere solo la parte economica.
run.py --no-score fa invece solo la parte costosa.

Le domande originali vengono rilete dal dataset (allineate per question_id), non assemblate dal file dei risultati — campi come rubric, meta
che servono per lo scoring non sono salvati nel file dei risultati.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import datasets                              # noqa: E402
from evaluation.run import make_llm, provenance, summarize   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Rivaluta risultati di valutazione già esistenti")
    p.add_argument("--file", required=True, help="JSON dei risultati prodotti da run.py")
    p.add_argument("--out", default="", help="Dove scrivere, sovrascrive --file di default")
    p.add_argument("--judge", default="", help="Modello judge, eredita quello della volta precedente di default")
    p.add_argument("--dataset", default="", choices=[""] + datasets.names(),
                   help="Eredità dai risultati del file di default")
    p.add_argument("--data", default="", help="Percorso del file dataset, eredità dai risultati del file di default")
    args = p.parse_args()

    src = Path(args.file)
    blob = json.loads(src.read_text(encoding="utf-8"))
    cfg = blob.get("config", {})

    dataset = args.dataset or cfg.get("dataset", "")
    data = args.data or cfg.get("data", "")
    judge_model = args.judge or cfg.get("judge", "gpt-4o-mini")
    if not dataset or not data:
        raise SystemExit("Il file dei risultati non ha registrato dataset/data, specifica con --dataset e --data")
    if not Path(data).exists():
        raise SystemExit(f"Il dataset non esiste più: {data}\nUsa --data per indicarne la posizione attuale")

    ds = datasets.get(dataset)
    # Indicizzazione per (id conversazione, id domanda): question_id è unico solo all'interno di una singola conversazione, collisioni tra segmenti
    # (LoCoMo ogni segmento ha q0/q1/…), usare solo q.id porterebbe a ottenere le risposte standard di un altro segmento.
    questions = {(c.id, q.id): q for c in ds.load(data) for q in c.questions}
    judge = make_llm(judge_model)

    n = sum(len(r["items"]) for r in blob["results"])
    print(f"Rivalutazione: {n} domande, judge {judge_model} (originale era {cfg.get('judge', '?')})", flush=True)

    changed, missing = 0, 0
    for r in blob["results"]:
        got = total = 0.0
        for it in r["items"]:
            q = questions.get((r["conversation_id"], it["question_id"]))
            if q is None:               # il dataset è cambiato, questa domanda non corrisponde — mantieni il punteggio originale e registra
                missing += 1
                got += it["correct"]
                total += it["total"]
                continue
            s = ds.score(q, it["predicted"], judge)
            if s.correct != it["correct"]:
                changed += 1
            it["correct"], it["total"], it["note"] = s.correct, s.total, s.note
            got += s.correct
            total += s.total
        r["score"], r["total"] = got, total

    blob["summary"] = summarize(blob["results"], dataset)
    blob["config"] = {**cfg, "judge": judge_model, "rescored": True}
    blob["provenance"] = provenance()

    out = Path(args.out or src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")

    s = blob["summary"]
    if missing:
        print(f"Attenzione: {missing} domande non trovate nel dataset, punteggio originale mantenuto")
    print(f"Riscritte {changed}/{n} domande")
    print(f"Punteggio {s['score']:.0f}/{s['total']:.0f}  =  {s['accuracy']:.1%}")
    print(f"Risultato salvato in {out}")


if __name__ == "__main__":
    main()
