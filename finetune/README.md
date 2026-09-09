# finetune

Addestra il tuo adapter di risposta VoiceMem.

```bash
pip install ms-swift==4.5.2 bitsandbytes    # installa torch secondo la tua piattaforma
python finetune/train.py            # prima fai girare la procedura con i 5 esempi inclusi
python finetune/train.py --data data/train.jsonl
```

Gli iperparametri default sono scritti in `finetune/utils.py` (`BASE` / `ADAPTER` / `TRAIN`), coerenti con
quella training dell'adapter pubblicato, **eseguire con default = riprodurre la stessa training** (base `Qwen/Qwen3.6-35B-A3B`,
LoRA rank 32 / alpha 64, 4bit).

## File

| | |
|---|---|
| `train.py` | Entry point training, `sft_main` di ms-swift |
| `eval.py` | Esegui l'adapter addestrato sui dati, stampa riga per riga `ref` / `pred` |
| `dataset.py` | Legge JSONL e **valida formato riga per riga**, se non conforme segnala direttamente quale riga e frase |
| `utils.py` | Iperparametri default, quattro system prompt, calcolo warmup |
| `data/sample.jsonl` | 5 esempi |

## Formato dati

```json
{
  "messages": [
    {"role": "system",    "content": "<seleziona in base a category/lang, vedi sotto>"},
    {"role": "user",      "content": "Come si chiama il mio gatto?\n\nMEMORY CONTEXT (things you remember about the user):\n- [2023-05-08] L'utente ha un gatto British Shorthair di nome Momo, tre anni."},
    {"role": "assistant", "content": "Si chiama Momo, un British Shorthair di tre anni."}
  ],
  "meta": {"lang": "zh", "category": "knowledge", "session_id": "s_0001", "turn": 1}
}
```
Il blocco memoria viene concatenato nell'**user dell'ultimo turno**, i turni storici non hanno memoria. Solo l'ultima frase assistant calcola il loss
(`loss_scale="last_round"`), i turni storici no.

## Parametri comuni

```bash
python finetune/train.py --data data/train.jsonl \
    --out out/my-adapter --epochs 3 --lr 1e-4 --no-4bit
```

| | | Default |
|---|---|---|
| `--data` | Dati training | `finetune/data/sample.jsonl` |
| `--out` | Directory output | `out/voicemem-qlora` |
| `--base` | Modello base | `Qwen/Qwen3.6-35B-A3B` |
| `--rank` / `--alpha` | Rango LoRA / alpha | 32 / 64 |
| `--epochs` / `--lr` | Epoch / learning rate | 2 / 2e-4 |
| `--max-len` | Lunghezza sequenza massima | 2048 |
| `--no-4bit` | Nessuna quantizzazione 4bit (usa solo se hai memoria VRAM sufficiente) | 4bit attivo di default |

**Cambiando il modello base devi modificare `target_regex`** — è `ADAPTER["target_modules"]` in `utils.py`,
scritto hardcoded secondo la denominazione dei moduli di Qwen3.6-35B-A3B. Se non sei sicuro, cambia con `all-linear`.

## Valutazione

```bash
python finetune/eval.py --adapter out/voicemem-qlora --out preds.jsonl
```

Stampa riga per riga `question` / `ref` / `pred` / `meta`, `--out` salva come JSONL.
Temperatura fissa a 0, utile per riprodurre e confrontare.

Per le metriche del retrieval memoria stesso vedi [`evaluation/`](../evaluation/).

## Note

- **I dati di training non sono in questo repository**. Prima della pubblicazione vanno aggiunti origine, licenza, stato del consenso e spiegazioni del preprocessing.
- L'adapter non può essere distribuito come modello indipendente, verifica autonomamente la licenza del modello base e le condizioni di accesso.
- Per il training multi-GPU `utils.warmup_steps()` è calcolato per singolo processo, va diviso ulteriormente per numero di GPU.
