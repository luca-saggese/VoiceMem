# VoiceMem EN/IT — Piano implementativo TTS locale con CosyVoice 3 Streaming

**Stato:** piano operativo per coding agent  
**Target:** fork VoiceMem EN/IT  
**TTS target:** `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`  
**Backend:** locale / self-hosted, nessuna chiamata OpenAI per sintesi vocale  
**Output VoiceMem:** PCM16 mono, 24 kHz  
**Lingue applicative:** italiano (`it`) e inglese (`en`)  
**Modalità CosyVoice target:** `inference_instruct2(..., stream=True)` con voce fissata da reference audio

---

# 1. Obiettivo

VoiceMem deve usare CosyVoice 3 come TTS predefinito al posto di OpenAI.

Pipeline target:

```text
LLM streaming text
        │
        ▼
sentence / phrase buffer
        │
        ▼
CosyVoice3TTS
        │
        ├── reference voice
        ├── language instruction EN/IT
        ├── emotion/prosody instruction
        └── stream=True
        │
        ▼
24 kHz float audio
        │
        ▼
PCM16 mono 24 kHz
        │
        ▼
VoiceMem websocket/playback
```

OpenAI può continuare a essere usato per chat/fact extraction se configurato, ma non deve essere necessario per TTS, generazione audio o playback.

---

# 2. Vincoli

L'implementazione deve:

- mantenere il contratto TTS esistente di VoiceMem;
- produrre **PCM16 mono 24 kHz**;
- produrre audio a chunk appena CosyVoice li rende disponibili;
- non attendere la generazione dell'intero WAV;
- non bloccare l'event loop asyncio;
- non caricare il modello a ogni frase;
- non caricare una copia del modello per ogni Memory Space;
- mantenere una voce coerente tra i turni;
- supportare italiano e inglese con lo stesso modello;
- mantenere il controllo prosodico/emozionale già presente in VoiceMem;
- interrompere rapidamente la riproduzione quando l'utente fa barge-in;
- non lasciare richieste TTS obsolete in coda dopo un'interruzione;
- non scrivere ogni frase sintetizzata su file temporaneo;
- non richiedere API key OpenAI per il TTS.

---

# 3. Stato attuale di VoiceMem

Il TTS è implementato principalmente in:

```text
voicemem/tts.py
```

Il contratto attuale è già adatto:

```python
class MyTTS:
    async def stream(self, text: str, instruction: str | None = None):
        # yield PCM16 mono 24k bytes
        ...
```

`BaseTTS` riceve `text`, riceve un'eventuale `instruction` per il singolo turno ed emette byte PCM16. Il web usa `memory_vm.utils.get("tts")` e poi `tts.stream(seg, speak_as)`, dove `speak_as` è già derivato dall'emozione del turno.

Questa interfaccia va mantenuta.

---

# 4. Problema importante: percorso web predefinito

VoiceMem Web supporta:

```text
realtime
llm_tts
```

Il default upstream è `realtime`, cioè voce OpenAI nativa. Se CosyVoice deve essere realmente il TTS principale, non basta aggiungere `provider=cosyvoice3`: il percorso standard deve diventare:

```text
LLM -> CosyVoice
```

Target:

```text
default mode = llm_tts
```

Decisione consigliata:

```text
llm_tts = default
realtime = opzionale oppure rimosso dopo validazione finale
```

Se l'obiettivo è **zero OpenAI per voce/audio**, rimuovere la modalità Realtime dopo che CosyVoice supera i gate finali.

---

# 5. Modello scelto

Usare:

```text
FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

Il modello supporta 9 lingue, inclusi italiano e inglese, audio-out streaming, zero-shot/cross-lingual voice cloning e instruction control. Il sample rate configurato nel checkpoint è **24 kHz**, quindi coincide con il contratto VoiceMem.

Repository modello:

```text
https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

Licenza dichiarata: Apache-2.0.

Directory proposta:

```text
models/
└── tts/
    └── Fun-CosyVoice3-0.5B-2512/
```

I pesi non devono essere committati.

---

# 6. Modalità di sintesi scelta

## Modalità principale: `inference_instruct2`

Target:

```python
model.inference_instruct2(
    tts_text=text,
    instruct_text=instruction,
    prompt_wav=reference_audio,
    stream=True,
    speed=speed,
)
```

Motivazione:

```text
reference audio -> chi sta parlando
instruction     -> come sta parlando
```

Questo si incastra bene con VoiceMem, che possiede già un'istruzione prosodica/emozionale per turno.

Non usare `cross_lingual` come modalità principale: è utile per cloning cross-language, ma non è il percorso migliore per il controllo dinamico della delivery.

---

# 7. Reference voice

Creare un profilo vocale stabile, ad esempio:

```text
voices/assistant.wav
```

Il reference deve essere:

- singolo parlante;
- pulito;
- senza musica;
- senza riverbero importante;
- senza clipping;
- volume stabile;
- parlato naturale per alcuni secondi.

Non cambiare reference a ogni turno.

La voce è proprietà della configurazione TTS, non della Memory Space. La stessa reference può essere usata per italiano e inglese.

---

# 8. Instruction CosyVoice 3

CosyVoice 3 richiede particolare attenzione al delimitatore:

```text
<|endofprompt|>
```

Il backend deve aggiungerlo automaticamente.

Esempio italiano:

```text
You are a helpful assistant. Speak in Italian. Parla con tono caldo, tranquillo e rassicurante.<|endofprompt|>
```

Esempio inglese:

```text
You are a helpful assistant. Speak in English. Speak calmly, warmly and reassuringly.<|endofprompt|>
```

Implementare una funzione equivalente a:

```python
def prepare_instruction(instruction: str | None, language: str) -> str:
    ...
```

I caller non devono gestire direttamente il token speciale.

---

# 9. Lingua EN/IT

La lingua della Memory Space resta la sorgente di verità:

```text
Space IT
    -> ASR it
    -> memory it
    -> reply it
    -> TTS "Speak in Italian"

Space EN
    -> ASR en
    -> memory en
    -> reply en
    -> TTS "Speak in English"
```

Non introdurre language autodetection nel TTS applicativo.

---

# 10. Struttura del codice

Scelta consigliata:

```text
voicemem/tts.py
    -> BaseTTS, factory, provider registry

voicemem/tts_cosyvoice.py
    -> CosyVoice3TTS
```

Non fare un refactor generale di tutti i provider TTS durante questo task.

Interfaccia target:

```python
class CosyVoice3TTS(BaseTTS):
    def __init__(
        self,
        model: str,
        ref_audio: str,
        *,
        default_language: str = "it",
        speed: float = 1.0,
        load_trt: bool = False,
        load_vllm: bool = False,
        fp16: bool = False,
    ):
        ...

    async def _raw(self, text: str, instruction: str | None = None):
        ...
```

Configurazione minima iniziale:

```text
model
ref_audio
default_language
speed
```

---

# 11. Lazy load e istanza condivisa

Il modello non deve essere caricato a import-time.

Implementare:

```python
def _load(self):
    if self._model is None:
        ...
    return self._model
```

La cache TTS già presente in VoiceMem deve essere sfruttata per ottenere:

```text
Space A ─┐
Space B ─┼──> shared CosyVoice3TTS
Space C ─┘
```

se la configurazione modello/voce è identica.

Lingua ed emozione devono essere informazioni per-request.

---

# 12. Integrazione source CosyVoice

## Non affidarsi al wheel PyPI `cosyvoice`

La distribuzione PyPI non è una base sufficientemente affidabile per l'inferenza completa. Il repository ufficiale usa il source tree e `third_party/Matcha-TTS`.

Strategia consigliata:

```text
third_party/
└── CosyVoice/
    ├── cosyvoice/
    ├── third_party/
    │   └── Matcha-TTS/
    └── ...
```

Usare git submodule o checkout pinning a una commit verificata.

Non seguire `main` senza pin in produzione.

Registrare:

```text
CosyVoice commit SHA
model revision SHA
```

in un file tipo:

```text
third_party/COSYVOICE_VERSION
```

---

# 13. Import path

CosyVoice necessita di Matcha-TTS nel path Python.

Non spargere `sys.path.append(...)` in più file.

Centralizzare la preparazione del path nel provider oppure nel launcher.

L'import effettivo deve rimanere lazy:

```python
from cosyvoice.cli.cosyvoice import AutoModel
```

solo quando il backend viene caricato.

---

# 14. Baseline di caricamento

Usare inizialmente:

```python
AutoModel(
    model_dir=model_path,
    load_trt=False,
    load_vllm=False,
    fp16=False,
)
```

Non attivare subito TensorRT, vLLM o fp16.

Prima ottenere correttezza e benchmark eager.

---

# 15. Nota critica TensorRT + fp16

Non usare come baseline:

```python
load_trt=True
fp16=True
```

Sono state segnalate configurazioni CosyVoice 3 con output non-finite/NaN usando TRT fp16.

Aggiungere comunque una guardia runtime:

```python
np.isfinite(audio).all()
```

Se fallisce, il chunk non deve essere inviato al client.

---

# 16. Conversione audio

CosyVoice restituisce `result["tts_speech"]` come tensore float.

Conversione target:

```python
audio = (
    result["tts_speech"]
    .detach()
    .float()
    .cpu()
    .numpy()
    .reshape(-1)
)

if not np.isfinite(audio).all():
    raise TTSInferenceError("CosyVoice returned non-finite audio")

audio = np.clip(audio, -1.0, 1.0)
pcm = (audio * 32767.0).astype(np.int16).tobytes()
```

Non usare WAV, MP3, AAC o Opus tra CosyVoice e VoiceMem.

Contratto interno:

```text
raw PCM16 mono 24000 Hz
```

---

# 17. Problema asyncio: il generatore CosyVoice è sincrono

Questo è un gate architetturale.

**Non fare:**

```python
async def _raw(...):
    for chunk in model.inference_instruct2(...):
        yield ...
```

L'inferenza PyTorch è sincrona e bloccherebbe l'event loop, compromettendo WebSocket, barge-in e altri task asyncio.

---

# 18. Worker dedicato

Architettura richiesta:

```text
async VoiceMem
     │
     ├── event loop
     │      │
     │      └── async audio consumer
     │
     └── CosyVoice worker thread
             │
             └── synchronous model generator
```

Il worker produce PCM e lo passa a una queue consumata dall'async generator.

La queue deve avere:

- backpressure o bound;
- sentinel di fine stream;
- propagazione delle eccezioni;
- cancellation flag;
- nessun busy loop.

---

# 19. Cancellation e barge-in

Quando l'utente interrompe:

1. fermare immediatamente il playback;
2. marcare la generazione CosyVoice come cancellata;
3. scartare ogni chunk successivo della richiesta vecchia;
4. evitare che segmenti non ancora iniziati partano;
5. liberare il modello appena possibile.

Usare cancellazione cooperativa:

```text
cancel_event.set()
```

Non tentare di terminare arbitrariamente kernel PyTorch in esecuzione.

---

# 20. Concorrenza

Il web upstream pre-lancia più segmenti TTS in parallelo per nascondere latenza HTTP. Questa strategia non è adatta a un unico modello CosyVoice locale.

Baseline:

```text
max concurrent CosyVoice generation = 1
```

Aggiungere una capability generica, per esempio:

```python
tts.max_concurrency = 1
```

oppure:

```python
tts.serial = True
```

Il web deve rispettarla.

Non hardcodare controlli `isinstance(CosyVoice3TTS)` nel web.

---

# 21. Segmentazione testo

Il chunking VoiceMem attuale è calibrato sulla latenza OpenAI TTS.

Non riutilizzare automaticamente:

```text
FIRST_MIN
FIRST_MAX
SENT_MIN
SENT_MAX
FIRST_SOFT
```

Baseline CosyVoice:

```text
FIRST_SOFT = false
```

Perché spezzare sulle virgole può introdurre reset prosodici e pause artificiali.

Se lo streaming CosyVoice ha TTFA buono, preferire frasi complete brevi e punteggiatura forte.

---

# 22. Bi-streaming CosyVoice 3

CosyVoice supporta input testuale via generator e output streaming. L'esempio ufficiale dimostra il concetto con `inference_zero_shot(text_generator(), ...)`.

Tuttavia VoiceMem vuole anche dynamic instruction via `inference_instruct2`.

**Non assumere senza test** che `inference_instruct2()` supporti il generator testuale con la stessa affidabilità.

Strategia:

### Fase 1 obbligatoria

```text
LLM streaming
 -> segmenti
 -> inference_instruct2(segment, stream=True)
 -> audio streaming
```

### Fase 2 sperimentale

Provare:

```python
inference_instruct2(
    text_generator(),
    instruction,
    ref_audio,
    stream=True,
)
```

Solo se supera gate di qualità, latenza, cancellation e assenza deadlock, usarlo in produzione.

---

# 23. Emotion/prosody instruction

Adattare:

```python
_speak_instruction(emotion)
```

in qualcosa tipo:

```python
_speak_instruction(emotion, language)
```

Esempio IT:

```text
Speak in Italian. Parla come un amico, con tono caldo. Rallenta leggermente e usa una voce morbida e rassicurante.
```

Esempio EN:

```text
Speak in English. Speak like a close friend, warmly and reassuringly. Slow down slightly and soften the voice.
```

Il backend aggiunge il wrapper CosyVoice e `<|endofprompt|>`.

---

# 24. Mapping emozionale

Non inviare solo label come `sad` o `happy`.

Preferire istruzioni descrittive.

## calm

```text
Parla con tono calmo e naturale, ritmo regolare, senza enfasi eccessiva.
```

## happy

```text
Parla con tono luminoso e leggermente energico, mantenendo naturalezza.
```

## sad/supportive

```text
Parla più lentamente, con tono basso, morbido e partecipe, senza teatralità.
```

## anxious user

```text
Mantieni un tono rassicurante e stabile; parla chiaramente e leggermente più lentamente.
```

L'emozione dell'utente non deve essere copiata meccanicamente nella voce dell'assistente.

---

# 25. Factory TTS

In `voicemem/tts.py` aggiungere:

```python
TTS_PROVIDERS = {
    ...
    "cosyvoice3": CosyVoice3TTS,
}
```

Default:

```python
TTS_BACKEND = os.environ.get("TTS_BACKEND", "cosyvoice3")
```

---

# 26. Config target

Esempio:

```python
"reply": {
    "llm": {
        "provider": "openai",
        "config": {"model": "..."},
    },
    "tts": {
        "provider": "cosyvoice3",
        "config": {
            "model": "models/tts/Fun-CosyVoice3-0.5B-2512",
            "ref_audio": "voices/assistant.wav",
            "default_language": "it",
            "speed": 1.0,
        },
    },
}
```

Questo consente LLM remoto ma voce completamente locale.

---

# 27. Warmup

CosyVoice ha un costo di caricamento importante.

Aggiungere warmup esplicito nel voice web mode:

```text
startup web
    ↓
load CosyVoice
    ↓
short discarded synthesis
    ↓
server ready
```

Non far pagare il model load al primo utente.

Misurare:

```text
model load time
warmup time
RSS / VRAM
```

---

# 28. Model download

Aggiornare:

```text
scripts/download_models.sh
```

Download:

```text
FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

in:

```text
models/tts/Fun-CosyVoice3-0.5B-2512
```

Usare `huggingface_hub.snapshot_download` con revision pin.

Baseline: snapshot completo.

Non ottimizzare `allow_patterns` prima di aver verificato quali file sono realmente necessari.

---

# 29. Dipendenze

Non riversare tutto `requirements.txt` CosyVoice nelle dipendenze core di VoiceMem.

Strategia consigliata:

```text
VoiceMem core dependencies
+
CosyVoice local setup
```

Aggiungere:

```text
scripts/setup_cosyvoice.sh
```

che:

1. inizializza/pinna il source CosyVoice;
2. inizializza Matcha-TTS;
3. installa requirements necessari;
4. verifica ambiente;
5. opzionalmente scarica il modello.

Python di riferimento iniziale:

```text
3.10
```

---

# 30. macOS / Apple Silicon

Questo è un gate tecnico.

L'implementazione ufficiale CosyVoice 3 disabilita TRT/fp16 se CUDA non è disponibile e non espone nel costruttore un backend MPS dedicato.

Quindi su Mac Apple Silicon **non assumere realtime**.

Misurare prima di integrare.

Metriche standalone:

```text
model load time
TTFA
audio duration
total inference time
RTF
peak RAM
```

Formula:

```text
RTF = inference_seconds / generated_audio_seconds
```

Target preferito:

```text
RTF <= 0.35
TTFA <= 500 ms
```

Gate minimo conversazionale:

```text
RTF < 1.0
TTFA <= 1200 ms
```

Se fallisce, non ottimizzare VoiceMem intorno a un TTS non realtime: usare CosyVoice come sidecar self-hosted su macchina GPU.

Rimane comunque zero OpenAI TTS.

---

# 31. Gate 0 — Standalone CosyVoice

Prima di modificare VoiceMem:

- [ ] source CosyVoice installato;
- [ ] Matcha-TTS disponibile;
- [ ] modello caricato;
- [ ] reference voice caricata;
- [ ] italiano sintetizzato;
- [ ] inglese sintetizzato;
- [ ] `stream=True` produce più chunk;
- [ ] sample rate = 24000;
- [ ] output finito;
- [ ] TTFA misurato;
- [ ] RTF misurato;
- [ ] RAM/VRAM misurata.

**Nessun commit VoiceMem se questo gate fallisce.**

---

# 32. STEP T1 — Provider CosyVoice3

File:

```text
voicemem/tts_cosyvoice.py
voicemem/tts.py
```

Implementare:

```text
CosyVoice3TTS
```

con:

```text
lazy model load
fixed ref audio
stream=True
float -> PCM16
finite validation
24k check
```

Non modificare ancora il web.

## Gate T1

Unit test con fake model:

- [ ] lazy load;
- [ ] model loaded once;
- [ ] output multiple chunks;
- [ ] PCM16 conversion;
- [ ] even byte chunks;
- [ ] NaN rejected;
- [ ] exception propagated;
- [ ] `instruction=None` funziona.

Eseguire:

```bash
python -m compileall voicemem
pytest -q
```

## Commit

```text
feat(tts): add CosyVoice3 streaming provider
```

---

# 33. STEP T2 — Async bridge

Spostare ogni inferenza CosyVoice fuori dall'event loop.

Aggiungere:

```text
worker
queue
exception sentinel
cancel event
```

## Gate T2

Durante una sintesi lunga eseguire in parallelo un ticker asyncio ogni 20–50 ms.

Il ticker deve rimanere responsivo.

## Commit

```text
feat(tts): run CosyVoice inference outside asyncio event loop
```

---

# 34. STEP T3 — Cancellation

Implementare cancellazione cooperativa.

## Gate T3

Sintetizzare 15–20 secondi, interrompere dopo circa 1 secondo e verificare:

- [ ] playback termina subito;
- [ ] nessun audio vecchio arriva al client;
- [ ] worker termina appena possibile;
- [ ] turno successivo parte;
- [ ] nessuna dead queue;
- [ ] nessun thread leak.

## Commit

```text
feat(tts): add cooperative cancellation for CosyVoice streams
```

---

# 35. STEP T4 — Serializzazione modello

Aggiungere policy:

```text
max_concurrency = 1
```

## Gate T4

Avviare due richieste artificialmente contemporanee.

Verificare:

```text
request A streams
request B waits/schedules
no corruption
no OOM
```

## Commit

```text
fix(tts): serialize local CosyVoice inference
```

---

# 36. STEP T5 — Provider default

Registrare `cosyvoice3` e renderlo default.

## Gate T5

`make_tts("cosyvoice3", ...)` deve restituire la stessa istanza per config identica.

Due Memory Space non devono duplicare il modello.

## Commit

```text
feat(tts): make CosyVoice3 the default TTS backend
```

---

# 37. STEP T6 — Lingua

Collegare lingua Space alla speaking instruction:

```text
IT -> Speak in Italian
EN -> Speak in English
```

## Gate T6

Test:

```text
IT
EN
IT -> EN -> IT
```

La voce deve restare la stessa.

## Commit

```text
feat(tts): bind CosyVoice speaking instructions to space language
```

---

# 38. STEP T7 — Emotion-driven delivery

Collegare l'emozione canonica VoiceMem alla speaking instruction.

Non cambiare il classifier emozionale in questo commit.

## Gate T7

Stesso testo con:

```text
neutral
happy
sad/supportive
calm
```

Verificare:

- [ ] stessa identità vocale;
- [ ] stile percepibilmente diverso;
- [ ] nessun cambio lingua;
- [ ] testo invariato;
- [ ] niente teatralità eccessiva.

## Commit

```text
feat(tts): drive CosyVoice prosody from turn emotion
```

---

# 39. STEP T8 — Web concurrency

Modificare:

```text
web/run.py
```

La pipeline deve rispettare `max_concurrency=1`.

## Gate T8

Risposta multi-frase:

- [ ] no overlap;
- [ ] no reorder;
- [ ] no OOM;
- [ ] nessuna frase vecchia dopo interrupt.

## Commit

```text
refactor(web): honor TTS concurrency capabilities
```

---

# 40. STEP T9 — Segmentazione CosyVoice

Ricalibrare chunking su misure reali.

Baseline:

```text
FIRST_SOFT = false
```

## Gate T9

Confronto A/B:

```text
old segmentation
new segmentation
```

Misurare:

```text
TTFA
inter-segment gap
number of TTS calls
prosodic continuity
```

## Commit

```text
perf(tts): tune text segmentation for CosyVoice streaming
```

---

# 41. STEP T10 — Warmup

Caricare e warmuppare CosyVoice allo startup del voice mode.

## Gate T10

Misurare cold vs warm first turn.

Il primo turno utente non deve pagare il model load.

## Commit

```text
perf(tts): warm CosyVoice before voice sessions
```

---

# 42. STEP T11 — Default web path

Cambiare default da:

```text
realtime
```

in:

```text
llm_tts
```

## Gate T11

```bash
python web/run.py
```

deve usare LLM text streaming + CosyVoice audio streaming.

Nessuna chiamata OpenAI audio.

## Commit

```text
feat(web): make local LLM plus TTS mode the default voice path
```

---

# 43. STEP T12 — Rimuovere OpenAI TTS

Rimuovere soltanto il backend speech OpenAI:

```text
OpenAITTS
OPENAI_TTS_VOICE
OPENAI_TTS_INSTRUCTIONS
OPENAI_TTS_BASE_URL
```

Non rimuovere OpenAI chat se ancora usato.

Audit:

```bash
rg -n 'OpenAITTS|OPENAI_TTS_|audio\.speech|TTS_BACKEND.*openai' voicemem web
```

## Gate T12

Nel percorso standard:

```text
0 OpenAI audio calls
```

## Commit

```text
refactor(tts): remove OpenAI speech backend
```

---

# 44. STEP T13 — Setup riproducibile

Aggiungere:

```text
scripts/setup_cosyvoice.sh
scripts/download_models.sh
```

Deve:

- inizializzare CosyVoice source;
- inizializzare Matcha-TTS;
- verificare Python;
- scaricare modello pinning revision;
- essere idempotente.

## Commit

```text
chore(tts): add reproducible CosyVoice setup and model download
```

---

# 45. STEP T14 — Cleanup config

Rimuovere configurazione OpenAI TTS obsoleta.

Il core deve continuare a fare:

```bash
python -c "import voicemem"
```

senza caricare CosyVoice.

## Commit

```text
chore(tts): remove obsolete OpenAI TTS configuration
```

---

# 46. STEP T15 — Spike bi-streaming

Solo dopo baseline stabile.

Testare:

```python
inference_instruct2(
    text_generator(),
    instruction,
    ref_audio,
    stream=True,
)
```

con segmenti che arrivano progressivamente.

## Gate T15

- [ ] input progressivo realmente consumato;
- [ ] audio parte prima della fine testo;
- [ ] instruction applicata;
- [ ] speaker stabile;
- [ ] no deadlock;
- [ ] cancellation funziona;
- [ ] qualità >= baseline;
- [ ] pause ridotte;
- [ ] TTFA non peggiora.

Se fallisce, mantenere per-sentence `inference_instruct2(..., stream=True)`.

---

# 47. STEP T16 — Bi-streaming produzione

Solo se T15 passa.

Aggiungere una API opzionale tipo:

```python
async def stream_text(self, segments, instruction=None):
    ...
```

senza rompere:

```python
stream(text, instruction)
```

Capability:

```text
supports_text_streaming = True
```

## Commit

```text
feat(tts): enable CosyVoice text-in audio-out bi-streaming
```

---

# 48. Gate barge-in finale

Scenario:

1. assistant genera risposta lunga;
2. audio parte;
3. utente interrompe;
4. playback si ferma;
5. ASR utente continua;
6. reply successivo parte;
7. CosyVoice vecchio non blocca il nuovo turno.

Misurare:

```text
audio stop latency
old generation cleanup latency
next TTS start latency
```

Target playback stop:

```text
<= 150 ms
```

Il vecchio audio non deve riapparire.

---

# 49. Benchmark finale

Corpus minimo:

```text
20 frasi italiane
20 frasi inglesi
```

Includere:

- saluti;
- date;
- numeri;
- nomi propri;
- inglesismi;
- frasi emotive;
- domande;
- frasi lunghe.

Metriche:

```text
TTFA
RTF
total latency
inter-segment gap
peak RAM
peak VRAM
chunk count
```

Qualità:

```text
pronuncia italiana
pronuncia inglese
stabilità speaker
prosodia
naturalità
assenza artefatti
assenza ripetizioni
assenza parole saltate
```

Test speciale:

```text
same text / different emotion instruction
IT -> EN -> IT
```

---

# 50. Failure isolation

Se CosyVoice fallisce:

- non crashare server;
- chiudere il segmento correttamente;
- loggare l'errore;
- non lasciare worker bloccati;
- permettere il turno successivo.

Non introdurre fallback OpenAI silenzioso.

Un eventuale fallback Piper/local è un task separato.

---

# 51. Logging e telemetria

Log utili:

```text
[tts] loading CosyVoice3...
[tts] ready model=... sr=24000
[tts] request lang=it chars=...
[tts] first_chunk_ms=...
[tts] rtf=...
[tts] cancelled
```

Metriche sviluppo:

```text
tts_load_ms
tts_first_audio_ms
tts_total_ms
tts_audio_ms
tts_rtf
tts_cancelled
tts_error
```

Non loggare reference voice bytes o contenuti privati completi.

---

# 52. Test unitari

Creare test per:

```text
provider factory
lazy loading
shared instance
instruction normalization
language instruction
PCM conversion
finite check
stream chunks
cancellation
exception propagation
model concurrency
```

Usare fake CosyVoice.

I test unitari non devono scaricare il modello.

---

# 53. Integration test reali

Separare con marker, per esempio:

```text
@pytest.mark.cosyvoice
@pytest.mark.slow
```

Test reale:

```text
models present
reference voice present
Italian sentence
English sentence
stream=True
```

Non includerlo nella suite leggera se il modello non è installato.

---

# 54. Criteri di accettazione finali

- [ ] `cosyvoice3` è provider TTS interno.
- [ ] CosyVoice 3 è il default TTS.
- [ ] modello: `Fun-CosyVoice3-0.5B-2512`.
- [ ] modello caricato una sola volta.
- [ ] stessa istanza condivisa tra Memory Space.
- [ ] output PCM16 mono 24 kHz.
- [ ] output a chunk.
- [ ] `stream=True` realmente usato.
- [ ] event loop non bloccato.
- [ ] inference fuori dal thread asyncio principale.
- [ ] cancellation funzionante.
- [ ] barge-in non lascia audio vecchio.
- [ ] massimo una inference baseline sul modello locale.
- [ ] italiano funzionante.
- [ ] inglese funzionante.
- [ ] voce coerente EN/IT.
- [ ] reference voice configurabile.
- [ ] instruction emozionale arriva a CosyVoice.
- [ ] `<|endofprompt|>` gestito automaticamente.
- [ ] NaN/non-finite audio rifiutato.
- [ ] OpenAI TTS non chiamato.
- [ ] default web path usa `llm_tts`.
- [ ] cold startup e warmup misurati.
- [ ] TTFA misurato.
- [ ] RTF misurato.
- [ ] RAM/VRAM misurata.
- [ ] model download ripetibile.
- [ ] pesi non tracciati in Git.
- [ ] test leggeri non richiedono modello.
- [ ] integration test reale passa.
- [ ] working tree finale pulito.

---

# 55. Commit history consigliata

```text
feat(tts): add CosyVoice3 streaming provider
feat(tts): run CosyVoice inference outside asyncio event loop
feat(tts): add cooperative cancellation for CosyVoice streams
fix(tts): serialize local CosyVoice inference
feat(tts): make CosyVoice3 the default TTS backend
feat(tts): bind CosyVoice speaking instructions to space language
feat(tts): drive CosyVoice prosody from turn emotion
refactor(web): honor TTS concurrency capabilities
perf(tts): tune text segmentation for CosyVoice streaming
perf(tts): warm CosyVoice before voice sessions
feat(web): make local LLM plus TTS mode the default voice path
refactor(tts): remove OpenAI speech backend
chore(tts): add reproducible CosyVoice setup and model download
chore(tts): remove obsolete OpenAI TTS configuration
```

Solo se lo spike bi-streaming passa:

```text
feat(tts): enable CosyVoice text-in audio-out bi-streaming
```

---

# 56. Cose da NON fare

Non:

```text
iterare CosyVoice direttamente nell'event loop
caricare il modello a ogni frase
caricare un modello per Memory Space
lanciare 3-4 inference simultanee sullo stesso modello
salvare ogni frase come WAV temporaneo
convertire 24k -> 48k -> 24k
attivare TRT + fp16 prima dei test di correttezza
assumere che macOS/MPS sia realtime senza benchmark
usare OpenAI come fallback silenzioso
cambiare ASR, emotion model e TTS nello stesso commit
sacrificare instruction-based prosody solo per dichiarare bi-streaming
```

---

# 57. Report finale richiesto all'agente

Alla fine produrre:

```text
1. commit creati
2. CosyVoice repo commit
3. model revision
4. modalità inference usata
5. reference voice path/config
6. sample rate verificato
7. model load time
8. warmup time
9. TTFA medio/p50/p95
10. RTF medio/p50/p95
11. peak RAM
12. peak VRAM se applicabile
13. barge-in stop latency
14. EN quality notes
15. IT quality notes
16. emotion instruction quality notes
17. test passati/falliti
18. eventuali issue aperte
19. esito spike bi-streaming
20. conferma zero chiamate OpenAI TTS
```

Non dichiarare completato se un gate obbligatorio fallisce.

---

# 58. Architettura target finale

```text
                    Memory Space
                    ┌───────────┐
                    │  IT / EN  │
                    └─────┬─────┘
                          │
                          ▼
                    Reply LLM stream
                          │
                          ▼
                   text segmentation
                          │
             ┌────────────┴────────────┐
             │                         │
             │ language                │ emotion
             ▼                         ▼
       Speak in Italian          supportive/calm/...
             │                         │
             └────────────┬────────────┘
                          ▼
                 CosyVoice instruction
                          │
                          ▼
                  CosyVoice3TTS
                  shared model
                  fixed voice
                          │
                          ▼
             inference_instruct2
                   stream=True
                          │
                          ▼
                float audio 24k
                          │
                          ▼
                 finite + clip
                          │
                          ▼
                 PCM16 mono 24k
                          │
                          ▼
             VoiceMem audio timeline
                          │
                          ▼
                 WebSocket/client
                          │
                          ▼
                       speaker
```

---

# 59. Riferimenti tecnici

CosyVoice repository:

https://github.com/FunAudioLLM/CosyVoice

CosyVoice 3 model:

https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

VoiceMem upstream:

https://github.com/xzf-thu/VoiceMem

VoiceMem TTS:

https://github.com/xzf-thu/VoiceMem/blob/main/voicemem/tts.py

CosyVoice example:

https://github.com/FunAudioLLM/CosyVoice/blob/main/example.py

CosyVoice implementation:

https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/cosyvoice.py

---

# 60. Decisione implementativa sintetica

La prima implementazione deve essere:

```text
CosyVoice 3 embedded/self-hosted
+
Fun-CosyVoice3-0.5B-2512
+
reference voice fissa
+
inference_instruct2
+
stream=True
+
worker dedicato
+
max concurrency 1
+
PCM16 24k diretto
+
EN/IT instruction
+
emotion/prosody instruction
```

Solo dopo benchmark separati valutare:

```text
bi-streaming text input
TensorRT
vLLM
fp16
multiple concurrent generations
```

Il criterio principale non è semplicemente "far partire CosyVoice", ma mantenere VoiceMem conversazionale:

```text
bassa TTFA
audio continuo
voce stabile
barge-in rapido
nessun blocco event loop
nessuna dipendenza OpenAI per la voce
```
