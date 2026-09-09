# VoiceMem EN/IT — Parte 2: integrazione emotion2vec+

## Missione

Integrare:

```text
iic/emotion2vec_plus_base
```

come classificatore acustico delle emozioni in VoiceMem.

Il modello deve:

* lavorare sull'audio dell'intera utterance;
* essere eseguito **dopo la chiusura VAD della frase**;
* non interferire con Nemotron streaming;
* non modificare il contratto ASR;
* non essere eseguito su ogni audio chunk;
* essere caricato una sola volta;
* funzionare sia per Memory Space `it` sia `en`;
* conservare gli score grezzi delle 9 classi;
* non forzare mapping semanticamente scorretti verso le emozioni canoniche VoiceMem.

Modello iniziale:

```text
iic/emotion2vec_plus_base
```

Non usare inizialmente:

```text
emotion2vec_plus_large
```

Il `base` è circa 90M parametri; il `large` circa 300M. Le versioni emotion2vec+ producono 9 classi con relativi score.

Classi:

```text
angry
disgusted
fearful
happy
neutral
other
sad
surprised
unknown
```

---

# Principio architetturale

La pipeline deve essere:

```text
                       ┌──> Nemotron
                       │      │
microfono -> VAD ------┤      └──> transcript
                       │
                       └──> buffer utterance
                              │
                              └──> emotion2vec
                                      │
                                      └──> acoustic emotion scores
```

Poi:

```text
transcript
+
acoustic emotion scores
+
eventuale emotion logic VoiceMem esistente
        │
        ▼
right-brain emotion attribution
```

ASR ed emotion recognition devono restare due capability indipendenti.

---

# Regola fondamentale

Non modificare:

```python
asr.feed()
asr.flush()
asr.reset()
```

per restituire emozioni.

Nemotron continua a restituire esclusivamente testo.

Emotion2vec riceve l'utterance audio separatamente.

---

# STEP E0 — Audit pipeline emozionale esistente

## Obiettivo

Prima di modificare codice, identificare esattamente come VoiceMem produce oggi:

```text
state.emotion
```

Eseguire:

```bash
rg -n 'emotion|Emotion|affect|prosody|anchor' voicemem web tests
```

Seguire il call path completo:

```text
audio
 -> perception
 -> emotion model
 -> attribution
 -> VoiceStream state
 -> right brain
 -> persistence
```

Identificare:

1. modello acustico esistente;
2. eventuale modello multimodale/LLM;
3. formato dell'output;
4. punto in cui viene valorizzato `state.emotion`;
5. punto in cui l'emozione viene passata al right brain;
6. eventuali score/confidence già disponibili;
7. eventuali dati persistiti.

Non sostituire alla cieca l'intera pipeline emozionale.

## GATE E0

L'agente deve essere in grado di descrivere:

```text
INPUT
  ↓
CURRENT EMOTION COMPONENT
  ↓
CANONICAL EMOTION
  ↓
RIGHT BRAIN
```

e identificare i file esatti coinvolti.

Nessun commit.

---

# STEP E1 — Introdurre il tipo `AcousticEmotionResult`

## Obiettivo

Separare chiaramente:

```text
acoustic emotion
```

da:

```text
canonical VoiceMem emotion
```

Aggiungere un tipo strutturato nel package audio/emotion appropriato.

Preferire una `dataclass`.

Esempio:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AcousticEmotionResult:
    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
```

Semantica:

```text
label = classe emotion2vec top-1
score = top score
scores = distribuzione completa
```

Non chiamare necessariamente il valore:

```text
confidence
```

perché gli score softmax del modello non devono essere considerati probabilità calibrate.

Preferire:

```text
score
top_score
```

---

# STEP E2 — Implementare `Emotion2VecClassifier`

## File

Preferire il package emotion già esistente.

Se esiste:

```text
voicemem/utils/audio/emotion/
```

creare:

```text
voicemem/utils/audio/emotion/emotion2vec.py
```

Se non esiste una struttura appropriata, usare:

```text
voicemem/utils/audio/emotion2vec.py
```

Non creare un nuovo sottosistema se non necessario.

---

## Contratto

Implementare:

```python
class Emotion2VecClassifier:
    def __init__(...):
        ...

    def classify(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> AcousticEmotionResult:
        ...
```

Il modello viene caricato nel costruttore.

`classify()` non deve caricare il modello.

---

## Modello

Configurazione:

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/emotion2vec_plus_base",
    hub="hf",
)
```

Per deployment fuori dalla Cina preferire `hf`; il progetto emotion2vec indica `hf`/`huggingface` come sorgente appropriata fuori dalla mainland China.

Non hardcodare `cuda`.

Il primo target è Mac CPU.

---

## Input

Emotion2vec deve ricevere:

```text
mono
float32
16000 Hz
```

Normalizzare l'array:

```python
samples = np.asarray(samples, dtype=np.float32).reshape(-1)
```

Se necessario effettuare resampling.

Se VoiceMem ha già una utility di resampling, riutilizzarla.

Non introdurre una seconda implementazione se non necessario.

---

## Niente WAV temporaneo

Usare direttamente:

```python
model.generate(
    input=samples_16k,
    granularity="utterance",
    extract_embedding=False,
)
```

FunASR accetta direttamente array NumPy audio a 16 kHz.

Non usare nel runtime:

```text
NamedTemporaryFile
sf.write(...)
temp.wav
```

---

## Parsing output

L'output atteso contiene:

```text
labels
scores
```

Le classi ufficiali sono nove.

Implementare parsing robusto:

```python
EXPECTED_LABELS = {
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
}
```

Non assumere l'ordine senza leggere `labels`.

Costruire:

```python
scores = {
    label: float(score)
    for label, score in zip(labels, raw_scores)
}
```

Poi:

```python
label = max(scores, key=scores.get)
score = scores[label]
```

---

## Error handling

Se il modello restituisce output invalido:

```python
AcousticEmotionResult(
    label="unknown",
    score=0.0,
    scores={},
)
```

oppure utilizzare il meccanismo di errore già adottato dal progetto.

Non fare crashare l'intero voice turn perché il classificatore emozionale ha fallito.

---

# GATE E2

Aggiungere unit test con mock FunASR:

```text
model loaded once
numpy input accepted
scores parsed correctly
top label selected
invalid output handled
unsupported/empty audio handled
```

Eseguire:

```bash
python -m compileall voicemem
pytest -q
```

Smoke test reale:

```text
neutral utterance
happy utterance
sad utterance
angry utterance
```

Non richiedere accuratezza perfetta.

Verificare soprattutto:

```text
nessun crash
9 score disponibili
label valida
score numerico
```

### Commit

```bash
git add voicemem/ tests/
git commit -m "feat(emotion): add emotion2vec acoustic classifier"
```

---

# STEP E3 — Gestione modello e warmup

## Obiettivo

Il modello deve essere creato una sola volta.

Architettura:

```text
VoiceMem startup
   ↓
Emotion2VecClassifier()
   ↓
AutoModel(...)
```

NON:

```text
utterance
   ↓
AutoModel(...)
```

---

## Warmup

Integrare nel sistema `warmup()` esistente.

Target:

```python
vm.warmup()
```

carica:

```text
Nemotron
VAD
voiceprint
emotion2vec
embedding
...
```

secondo le capability abilitate.

Non caricare emotion2vec in modalità:

```text
leftbrain_only
```

se l'audio emotion non viene utilizzato.

---

# GATE E3

Aggiungere instrumentation/test che dimostri:

```text
20 utterance
AutoModel constructed = 1 volta
classify called = 20 volte
```

Controllare RAM:

```text
warmup
utterance 1
utterance 10
utterance 20
```

Non deve esserci crescita continua evidente.

### Commit

```bash
git add voicemem/ tests/
git commit -m "feat(emotion): integrate emotion2vec lifecycle and warmup"
```

---

# STEP E4 — Conservare l'audio della singola utterance

## Obiettivo

Emotion2vec deve analizzare esattamente l'audio appartenente al turno appena concluso.

Durante il voice turn mantenere un buffer:

```text
utterance_audio
```

Il buffer:

* inizia quando il VAD apre il turno;
* accumula audio parlato;
* termina quando il VAD chiude il turno;
* viene passato una volta a emotion2vec;
* viene azzerato prima del turno successivo.

Non usare l'intera registrazione della sessione.

---

## Pre-roll

Se VoiceMem VAD usa pre-roll, includerlo.

Non tagliare la prima sillaba.

---

## Tail silence

È possibile rimuovere parte del silenzio finale prima di emotion2vec.

Non è obbligatorio nella prima versione.

Non introdurre algoritmo di trimming complesso.

---

## Limite durata

Aggiungere una protezione ragionevole.

Esempio:

```text
max emotion utterance = 30–60 s
```

Se un turno supera il limite:

* usare la porzione appropriata;
* oppure saltare emotion inference;
* documentare la scelta.

Non permettere accumulo audio senza limite.

---

# GATE E4

Testare:

```text
utterance A
silenzio
utterance B
```

Verificare:

```text
emotion A usa solo audio A
emotion B usa solo audio B
```

Testare anche:

```text
reset
cancel
space switch
exception
```

Il buffer deve sempre essere rilasciato.

### Commit

```bash
git add voicemem/stream.py voicemem/ tests/
git commit -m "feat(emotion): collect utterance audio for acoustic inference"
```

---

# STEP E5 — Integrare emotion2vec al turn-over

## Obiettivo

Eseguire emotion inference solamente dopo che il VAD determina:

```text
turn_over
```

Sequenza target:

```text
VAD endpoint
    ↓
Nemotron.flush()
    ↓
final transcript
    ↓
finalize utterance audio
    ↓
emotion2vec.classify()
    ↓
acoustic emotion result
    ↓
right-brain attribution
```

Non chiamare emotion2vec:

```text
ogni 32 ms
ogni 100 ms
su ogni ASR partial
```

---

## Regola latency

Nemotron deve continuare a produrre partial durante il parlato senza aspettare emotion2vec.

Emotion inference avviene solo al termine.

Il codice emotion non deve essere nel callback audio CoreAudio.

---

# GATE E5

Con instrumentation verificare:

Durante la frase:

```text
ASR partial
ASR partial
ASR partial
```

solo dopo endpoint:

```text
ASR final
emotion inference
```

Mai:

```text
emotion inference
ASR partial
emotion inference
ASR partial
```

### Commit

```bash
git add voicemem/stream.py voicemem/
git commit -m "feat(emotion): run acoustic inference at end of utterance"
```

---

# STEP E6 — Esporre il risultato acustico nello Stream State

## Obiettivo

Non sovrascrivere subito:

```python
state.emotion
```

con la label raw emotion2vec.

Aggiungere, se compatibile con l'API corrente:

```python
state.acoustic_emotion
state.acoustic_emotion_score
state.acoustic_emotion_scores
```

Esempio:

```python
state.acoustic_emotion = "sad"
state.acoustic_emotion_score = 0.68
state.acoustic_emotion_scores = {
    "sad": 0.68,
    "neutral": 0.17,
    ...
}
```

L'attuale:

```python
state.emotion
```

deve continuare a rappresentare l'emozione canonica VoiceMem.

Questo evita di rompere il right brain.

---

# GATE E6

Testare che:

```text
state.emotion
```

mantenga il precedente contratto.

E:

```text
state.acoustic_emotion*
```

contenga emotion2vec.

Nessuna API esistente deve cambiare significato silenziosamente.

### Commit

```bash
git add voicemem/ tests/
git commit -m "feat(emotion): expose acoustic emotion scores in stream state"
```

---

# STEP E7 — Non fare mapping 9 → 8 ingenuo

## Problema

Emotion2vec:

```text
angry
disgusted
fearful
happy
neutral
other
sad
surprised
unknown
```

VoiceMem usa un'ontologia differente, ad esempio:

```text
anxious
sad
wronged
lonely
conflicted
calm
happy
tired
```

Non esiste una corrispondenza 1:1.

Esempi sbagliati:

```text
angry -> wronged
disgusted -> conflicted
surprised -> happy
neutral -> calm
```

Sono inferenze semanticamente scorrette.

---

# Strategia corretta

Conservare due livelli:

```text
ACOUSTIC EMOTION
    emotion2vec raw

CANONICAL EMOTION
    VoiceMem right-brain
```

Poi usare emotion2vec come evidenza.

---

# STEP E8 — Implementare il fusion layer

## Obiettivo

Creare un layer piccolo e testabile.

Esempio:

```python
def attribute_emotion(
    transcript: str,
    acoustic: AcousticEmotionResult | None,
    ...
) -> CanonicalEmotion:
    ...
```

Il fusion layer può utilizzare:

```text
transcript
emotion2vec scores
eventuale classifier VoiceMem esistente
context
```

Non nascondere il mapping dentro `Emotion2VecClassifier`.

---

## Prima versione: mapping conservativo

Consentire solo mapping forti.

Esempio iniziale:

```text
happy    -> happy
sad      -> sad
fearful  -> anxious
```

Con cautela:

```text
neutral -> nessuna evidenza forte
```

Non forzare:

```text
angry
disgusted
surprised
other
unknown
```

su categorie VoiceMem non equivalenti.

Questi diventano:

```text
acoustic evidence only
```

---

## Threshold

Non usare:

```python
if score > 0.5:
```

come verità universale senza benchmark.

Introdurre una costante configurabile:

```python
ACOUSTIC_EMOTION_MIN_SCORE
```

ma determinare il valore iniziale dal corpus EN/IT.

Fino ad allora usare il risultato come evidenza, non decisione assoluta.

---

# GATE E8

Testare almeno:

```text
happy high score
sad high score
fearful high score
angry high score
neutral high score
unknown high score
```

Verificare che:

```text
happy -> happy
sad -> sad
fearful -> anxious
```

e che:

```text
angry
disgusted
surprised
```

non vengano convertite arbitrariamente.

### Commit

```bash
git add voicemem/rightbrain/ voicemem/ tests/
git commit -m "feat(emotion): fuse acoustic emotion with right brain attribution"
```

---

# STEP E9 — Fallback e failure isolation

## Obiettivo

VoiceMem deve continuare a funzionare anche se emotion2vec:

```text
non è disponibile
fallisce il download
fallisce inference
riceve audio invalido
va out of memory
```

Comportamento:

```text
ASR continua
retrieval continua
memory continua
reply continua
```

Il risultato emozionale diventa:

```text
unknown / None
```

Non abortire il turno.

---

## Logging

Loggare una volta il problema con contesto sufficiente.

Non stampare stack trace ad ogni utterance se il modello rimane indisponibile.

---

# GATE E9

Simulare:

```text
model unavailable
classify raises RuntimeError
empty audio
NaN audio
```

Verificare che il turno produca comunque:

```text
transcript
retrieval
reply
```

### Commit

```bash
git add voicemem/ tests/
git commit -m "fix(emotion): isolate acoustic emotion inference failures"
```

---

# STEP E10 — Benchmark prestazionale su Mac

## Corpus

Almeno:

```text
20 utterance IT
20 utterance EN
```

Durate:

```text
1–2 s
3–5 s
8–12 s
```

Misurare:

```text
model load time
RAM
inference time
RTF
turn-over latency
```

Calcolare:

```text
RTF = inference_seconds / audio_seconds
```

---

## Gate prestazionale

Richiesto:

```text
RTF < 1.0
```

Emotion inference deve essere più veloce della durata dell'audio.

Preferibile:

```text
RTF < 0.25
```

Se:

```text
RTF >= 1
```

non integrare emotion inference in modo sincrono nel percorso critico senza ulteriore ottimizzazione.

Misurare, non indovinare.

---

## Gate memoria

Verificare:

```text
Nemotron loaded
emotion2vec loaded
20 turns
```

La RAM non deve crescere continuamente turno dopo turno.

Una crescita iniziale dovuta alle cache PyTorch è accettabile.

Una crescita lineare per utterance non lo è.

---

# STEP E11 — Corpus qualitativo italiano/inglese

Emotion2vec non deve essere accettato perché "sembra funzionare".

Creare un piccolo benchmark manuale.

Usare la stessa frase con delivery differente.

Italiano:

```text
"Oggi sono andato a comprare il pane."
```

Inglese:

```text
"Today I went to buy some bread."
```

Registrare:

```text
neutral
happy
sad
angry
fearful
surprised
```

Idealmente più di un parlante.

---

## Metriche

Per ogni file salvare:

```text
expected
top1
top1_score
top3
full scores
```

Non modificare/tarare il sistema dopo ogni singolo errore.

Guardare pattern aggregati.

---

# GATE E11

Il modello non deve necessariamente essere eccellente.

Deve però dimostrare:

1. output non costante;
2. differenziazione sensata di almeno alcune emozioni;
3. comportamento simile fra EN e IT;
4. nessun forte bias sistematico verso una singola classe;
5. score utili come segnale secondario.

Se produce ad esempio:

```text
neutral
neutral
neutral
neutral
neutral
```

sulla quasi totalità del corpus, non integrarlo nel right brain.

Può comunque essere lasciato come capability sperimentale.

### Commit

```bash
git add tests/ evaluation/
git commit -m "test(emotion): add Italian and English acoustic emotion benchmark"
```

Se `evaluation/` non è appropriato nel fork, usare una directory test dedicata già esistente.

---

# STEP E12 — Model download

## Obiettivo

Il primo avvio dell'applicazione non deve sorprendere l'utente scaricando centinaia di MB senza indicazione.

Integrare emotion2vec nella procedura modelli.

Dato che FunASR/Hugging Face gestisce il formato del modello, non copiare manualmente file individuali se non necessario.

Preferire un piccolo script Python di prefetch richiamato da:

```text
scripts/download_models.sh
```

Esempio concettuale:

```python
from funasr import AutoModel

AutoModel(
    model="iic/emotion2vec_plus_base",
    hub="hf",
)
```

Oppure usare un percorso locale se il progetto decide di consolidare tutti i modelli in `models/emotion/`.

Scegliere una sola strategia.

---

# GATE E12

Su ambiente senza cache:

```bash
bash scripts/download_models.sh
```

Poi disabilitare rete e verificare che emotion2vec venga caricato dalla cache/path locale.

Nessun checkpoint deve essere tracked da Git.

Verificare:

```bash
git status
```

### Commit

```bash
git add scripts/
git commit -m "chore(models): add emotion2vec model prefetch"
```

---

# STEP E13 — Dipendenze

## File

```text
pyproject.toml
```

Emotion2vec tramite FunASR richiede di mantenere:

```text
funasr
torch
```

e le dipendenze necessarie al caricamento del modello.

Quindi il precedente task:

```text
remove FunASR
```

va annullato/omesso se emotion2vec viene adottato.

Non mantenere FunASR come ASR.

Deve essere chiaro che:

```text
sherpa-onnx -> ASR
FunASR      -> emotion2vec loader/inference
```

---

# GATE E13

Installazione da zero:

```bash
python -m venv /tmp/voicemem-emotion
source /tmp/voicemem-emotion/bin/activate

python -m pip install -U pip
pip install -e .
```

Poi:

```bash
python -c "from funasr import AutoModel"
python -c "import sherpa_onnx"
python -c "import voicemem"
```

Smoke:

```text
Nemotron IT
Nemotron EN
emotion2vec
```

### Commit

Solo se necessario:

```bash
git add pyproject.toml
git commit -m "chore(deps): retain FunASR for emotion2vec inference"
```

---

# STEP E14 — UI debug opzionale

Non mostrare necessariamente la classificazione emozionale all'utente finale.

Per sviluppo può essere utile una modalità debug:

```text
Emotion: triste
Acoustic: sad 0.71
```

con distribuzione eventualmente disponibile in console/debug panel.

Non trasformare l'interfaccia normale in un dashboard di classificazione.

---

# GATE E14

La UI normale deve continuare a essere pulita.

Debug:

```text
off -> nessun dettaglio tecnico
on  -> label + score disponibili
```

### Commit

Solo se implementato:

```bash
git add web/
git commit -m "feat(debug): expose acoustic emotion diagnostics"
```

---

# STEP E15 — Gate finale emotion2vec

Eseguire:

```bash
python -m compileall voicemem web
pytest -q
```

Poi smoke test completo:

```text
IT Space
 ↓
Italian speech
 ↓
Nemotron partial
 ↓
Nemotron final
 ↓
emotion2vec scores
 ↓
canonical emotion
 ↓
right brain
 ↓
reply italiano
```

Poi:

```text
EN Space
 ↓
English speech
 ↓
Nemotron partial
 ↓
Nemotron final
 ↓
emotion2vec scores
 ↓
canonical emotion
 ↓
right brain
 ↓
reply English
```

---

# Acceptance gate

Tutti devono essere veri:

* [ ] emotion2vec usa `iic/emotion2vec_plus_base`.
* [ ] Il modello viene caricato una sola volta.
* [ ] L'inferenza avviene a livello utterance.
* [ ] Non viene eseguita su ogni audio chunk.
* [ ] Nemotron streaming non dipende da emotion2vec.
* [ ] ASR continua a funzionare se emotion2vec fallisce.
* [ ] L'audio viene passato in mono float32 16 kHz.
* [ ] Non vengono creati WAV temporanei per ogni turno.
* [ ] Sono conservati label e score raw.
* [ ] Sono conservati tutti gli score delle 9 classi.
* [ ] Il top score non viene trattato come probabilità calibrata.
* [ ] Il risultato raw non sostituisce direttamente l'emozione canonica VoiceMem.
* [ ] Il mapping 9→8 non è arbitrario.
* [ ] `angry`, `surprised`, `disgusted` non vengono forzati in categorie non equivalenti.
* [ ] Il fusion layer è separato dal classifier.
* [ ] Italiano e inglese usano lo stesso modello emotion2vec.
* [ ] emotion2vec non usa la lingua della Memory Space come parametro.
* [ ] Cambio Space non richiede ricaricare emotion2vec.
* [ ] `RTF < 1` sul Mac target.
* [ ] Nessuna crescita RAM lineare su 20+ turni.
* [ ] Un errore emotion non interrompe il voice turn.
* [ ] FunASR rimane dipendenza esclusivamente dove necessario.
* [ ] Test qualitativi EN/IT sono documentati.
* [ ] Working tree finale pulito.

---

# Commit history attesa

```text
feat(emotion): add emotion2vec acoustic classifier

feat(emotion): integrate emotion2vec lifecycle and warmup

feat(emotion): collect utterance audio for acoustic inference

feat(emotion): run acoustic inference at end of utterance

feat(emotion): expose acoustic emotion scores in stream state

feat(emotion): fuse acoustic emotion with right brain attribution

fix(emotion): isolate acoustic emotion inference failures

test(emotion): add Italian and English acoustic emotion benchmark

chore(models): add emotion2vec model prefetch
```

Eventualmente:

```text
chore(deps): retain FunASR for emotion2vec inference
```

e solo se realmente utile:

```text
feat(debug): expose acoustic emotion diagnostics
```

---

# Cose da NON fare

Non:

```text
sostituire Nemotron
```

Non:

```text
far passare emotion2vec dentro l'ASR
```

Non:

```text
eseguire emotion2vec sui partial audio
```

Non:

```text
ricaricare emotion2vec a ogni frase
```

Non:

```text
usare emotion2vec_plus_large prima del benchmark
```

Non:

```text
considerare score=0.8 come confidence calibrata 80%
```

Non:

```text
mappare angry -> wronged
```

Non:

```text
mappare surprised -> happy
```

Non:

```text
mappare neutral -> calm automaticamente
```

Non:

```text
bloccare ASR/reply se emotion2vec fallisce
```

Non:

```text
scrivere ogni utterance su disco per poterla classificare
```

Non:

```text
modificare contemporaneamente l'ontologia emozionale interna
```

---

# Architettura target finale

```text
                         ┌──────────────────┐
                         │   Memory Space   │
                         │      IT / EN     │
                         └────────┬─────────┘
                                  │
                           language for ASR
                                  │
                                  ▼
MIC ──> VAD ───────────────> Nemotron 3.5
 │        │                       │
 │        │                       ├── partial
 │        │                       └── final transcript
 │        │
 │        └── utterance buffer
 │                 │
 │                 ▼
 │       emotion2vec+ base
 │                 │
 │                 ▼
 │       raw acoustic scores
 │                 │
 │                 ├── angry
 │                 ├── disgusted
 │                 ├── fearful
 │                 ├── happy
 │                 ├── neutral
 │                 ├── other
 │                 ├── sad
 │                 ├── surprised
 │                 └── unknown
 │
 └─────────────────────────────────────────┐
                                           │
                  transcript + acoustic evidence
                                           │
                                           ▼
                                  emotion attribution
                                           │
                                           ▼
                               VoiceMem canonical emotion
                                           │
                                           ▼
                                      Right Brain
```

Il concetto da preservare è:

```text
Nemotron = cosa ha detto

emotion2vec = come suonava la voce

right brain = cosa significa emotivamente nel contesto
```

I tre livelli non devono essere fusi nello stesso componente.

---

# Report finale richiesto all'agente

Alla fine produrre:

```text
1. commit creati
2. file modificati
3. punto esatto della pipeline dove gira emotion2vec
4. modello e hub utilizzato
5. model load time
6. RAM dopo warmup
7. RTF medio
8. RTF p95
9. risultati benchmark italiano
10. risultati benchmark inglese
11. confusioni più comuni osservate
12. failure/fallback testati
13. mapping applicati verso le canonical emotions
14. classi lasciate volutamente non mappate
15. problemi aperti
```

Non dichiarare l'integrazione conclusa se:

```text
emotion2vec blocca Nemotron
```

oppure:

```text
il modello viene caricato a ogni turno
```

oppure:

```text
gli errori emotion fanno fallire il turno
```

oppure:

```text
RTF >= 1 sul Mac target
```
