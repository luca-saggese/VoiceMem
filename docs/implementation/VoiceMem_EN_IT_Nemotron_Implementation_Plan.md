# VoiceMem EN/IT — Piano implementativo
## Migrazione ASR a Nemotron 3.5 Streaming 0.6B e rimozione del supporto cinese

**Stato:** piano di implementazione  
**Target:** fork di VoiceMem non compatibile con upstream, semplificato per **Italiano + Inglese**  
**ASR target:** `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11`  
**Lingue supportate:** `it`, `en`  
**Lingua predefinita proposta:** `it`  
**Principio:** una Memory Space ha una lingua fissa; ASR, memoria e risposta usano la lingua della Space.

---

# 1. Obiettivo

Il fork deve:

1. sostituire i backend ASR attuali con **Nemotron 3.5 ASR Streaming 0.6B**;
2. usare **un solo modello ASR multilingual** per italiano e inglese;
3. mantenere il comportamento streaming e i partial transcript di VoiceMem;
4. conservare il contratto ASR già usato da `VoiceStream`:
   - `feed(samples) -> testo cumulativo`
   - `flush() -> testo finale`
   - `reset()`
5. eliminare il cinese come lingua supportata;
6. supportare soltanto:
   - `it`
   - `en`
7. mantenere la lingua come proprietà della **Memory Space**;
8. forzare la lingua di Nemotron per ogni stream:
   - `it` per Memory Space italiana;
   - `en` per Memory Space inglese;
9. evitare `language="auto"` nel normale funzionamento;
10. mantenere ASR ed emotion recognition come capacità separate.

Non è un obiettivo mantenere la compatibilità con il progetto VoiceMem originale.

---

# 2. Stato attuale rilevante di VoiceMem

## 2.1 ASR

Il file principale è:

```text
voicemem/utils/audio/asr.py
```

Attualmente contiene due implementazioni streaming:

```text
FunASRStreamingASR
    -> paraformer-zh-streaming

StreamingASR
    -> sherpa-onnx Zipformer bilingual zh/en
```

La factory è in:

```text
voicemem/utils/defaults.py
```

e seleziona il backend usando:

```text
VOICEMEM_ASR
```

Il default attuale è FunASR/Paraformer cinese.

Per il fork EN/IT questa architettura non serve più: entrambi i backend possono essere sostituiti da un'unica implementazione Nemotron.

---

## 2.2 Contratto richiesto da `VoiceStream`

`voicemem/stream.py` utilizza l'ASR in questo modo:

```python
self._text = self.asr.feed(frame)
```

e alla fine del turno:

```python
self._text = self._asr.flush() or self._text
```

poi esegue:

```python
self._asr.reset()
```

Questa è una buona interfaccia e va mantenuta.

**Non modificare la logica di speculative retrieval di `VoiceStream`.**

La migrazione ASR deve essere contenuta quasi completamente nel provider ASR.

---

# 3. Architettura target

```text
                     Memory Space
                         │
                    language
                    ┌────┴────┐
                    │         │
                   "it"      "en"
                    │         │
                    └────┬────┘
                         │
                         ▼
              NemotronStreamingASR
                         │
               stream.set_option(
                 "language", lang
               )
                         │
                         ▼
microfono -> resample 16 kHz -> Nemotron -> partial transcript
       │
       └-------------------------------> VAD / audio perception
```

Nemotron non sostituisce:

- Silero VAD;
- voiceprint;
- scene detection;
- emotion detection;
- embedding;
- memoria;
- retrieval.

Sostituisce soltanto l'ASR.

---

# 4. Modello ASR scelto

Usare inizialmente:

```text
sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
```

File richiesti:

```text
encoder.int8.onnx
decoder.int8.onnx
joiner.int8.onnx
tokens.txt
```

Layout proposto:

```text
models/
└── asr/
    └── sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/
        ├── encoder.int8.onnx
        ├── decoder.int8.onnx
        ├── joiner.int8.onnx
        └── tokens.txt
```

Il modello supporta nativamente italiano e inglese.

La lingua viene impostata per stream:

```python
stream.set_option("language", "it")
```

oppure:

```python
stream.set_option("language", "en")
```

Nel fork **non usare auto-detection come default**.

Motivazione:

- la lingua è già nota dalla Memory Space;
- evita errori sui turni molto brevi;
- evita cambi di lingua accidentali;
- rende il risultato più ripetibile;
- semplifica test e debugging.

---

# 5. Fase 1 — Implementare `NemotronStreamingASR`

## File

```text
voicemem/utils/audio/asr.py
```

## Strategia

Rimuovere o dismettere:

```python
FunASRStreamingASR
StreamingASR
```

e introdurre:

```python
NemotronStreamingASR
```

Non è necessario mantenere i vecchi nomi se il fork non deve essere compatibile con upstream.

---

## 5.1 Interfaccia proposta

```python
class NemotronStreamingASR:
    def __init__(
        self,
        asr_dir: str,
        language: str,
        *,
        num_threads: int = 4,
        provider: str = "cpu",
    ):
        ...

    def feed(self, samples) -> str:
        ...

    def flush(self) -> str:
        ...

    def reset(self) -> None:
        ...
```

Le uniche lingue valide devono essere:

```python
SUPPORTED_ASR_LANGUAGES = ("it", "en")
```

---

## 5.2 Implementazione di riferimento

```python
from __future__ import annotations

from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
SUPPORTED_ASR_LANGUAGES = ("it", "en")


class NemotronStreamingASR:
    def __init__(
        self,
        asr_dir: str,
        language: str,
        *,
        num_threads: int = 4,
        provider: str = "cpu",
    ) -> None:
        import sherpa_onnx

        lang = str(language).strip().lower()

        if lang not in SUPPORTED_ASR_LANGUAGES:
            raise ValueError(
                f"Unsupported ASR language {language!r}; "
                f"expected one of {SUPPORTED_ASR_LANGUAGES}"
            )

        model_dir = Path(asr_dir)

        self.language = lang
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_dir / "tokens.txt"),
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
        )

        self.stream = None
        self.reset()

    def _new_stream(self):
        stream = self.rec.create_stream()
        stream.set_option("language", self.language)
        return stream

    @staticmethod
    def _text(result) -> str:
        if isinstance(result, str):
            return result.strip()

        return str(getattr(result, "text", result) or "").strip()

    def feed(self, samples) -> str:
        frame = np.asarray(samples, dtype=np.float32)

        self.stream.accept_waveform(SAMPLE_RATE, frame)

        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)

        return self._text(self.rec.get_result(self.stream))

    def flush(self) -> str:
        self.stream.input_finished()

        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)

        return self._text(self.rec.get_result(self.stream))

    def reset(self) -> None:
        self.stream = self._new_stream()
```

---

## 5.3 Nota importante su `flush()`

Nemotron/sherpa usa:

```python
stream.input_finished()
```

per segnalare che non arriverà altro audio su quello stream.

Questo è compatibile con il flusso attuale di VoiceMem perché:

```text
VAD conferma fine turno
        ↓
ASR.flush()
        ↓
_turn_over
        ↓
ASR.reset()
```

Dopo `flush()` non bisogna riutilizzare lo stesso sherpa stream.

`reset()` deve sempre creare un nuovo stream e riapplicare:

```python
stream.set_option("language", self.language)
```

---

# 6. Fase 2 — Rendere Nemotron il solo backend ASR

## File

```text
voicemem/utils/defaults.py
```

La factory attuale contiene la selezione:

```text
VOICEMEM_ASR=funasr
VOICEMEM_ASR=sherpa
```

Per il fork questa biforcazione va eliminata.

Target:

```python
def asr():
    from voicemem.lang import memory_language
    from voicemem.utils.audio.asr import NemotronStreamingASR
    from voicemem.utils.common.paths import model_path

    lang = memory_language()

    return NemotronStreamingASR(
        str(
            model_path(
                "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
                kind="asr",
            )
        ),
        language=lang,
    )
```

Questo mantiene una caratteristica utile dell'architettura corrente:

- la factory ASR viene caricata lazy;
- `VoiceMem` ha già risolto la lingua della Memory Space prima che inizi lo streaming;
- l'ASR viene costruito usando la lingua della Space.

---

## 6.1 Eliminare `VOICEMEM_ASR`

Se non serve compatibilità:

```text
VOICEMEM_ASR
```

va eliminato.

Non serve poter selezionare:

```text
funasr
sherpa
nemotron
```

se Nemotron è l'unico ASR supportato.

Questo evita una configurazione inutile.

---

# 7. Fase 3 — Download del modello

## File

```text
scripts/download_models.sh
```

Attualmente il progetto scarica:

```text
funasr-paraformer-zh-streaming
sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
```

Entrambi possono essere eliminati dal bundle ASR.

Definire:

```bash
ASR_DIR="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
```

Download:

```bash
REL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

mkdir -p "${DEST}/asr"

if [ ! -d "${DEST}/asr/${ASR_DIR}" ]; then
    curl -L \
      "${REL}/${ASR_DIR}.tar.bz2" \
      | tar xj -C "${DEST}/asr"
fi
```

Non scaricare più:

```text
paraformer-zh-streaming
zipformer-bilingual-zh-en
```

---

## 7.1 Bundle modelli

Il repository upstream usa anche un bundle Hugging Face:

```text
zhifeixie/VoiceMem_Default_Models_Env
```

Per il fork non conviene dipendere da quel bundle per l'ASR.

Strategia consigliata iniziale:

1. continuare a usare il bundle esistente per gli altri modelli, se serve;
2. scaricare Nemotron direttamente dalla release ufficiale sherpa-onnx;
3. in seguito creare eventualmente un bundle modelli proprio.

Questo evita di dover pubblicare subito un repository di modelli del fork.

---

# 8. Fase 4 — Dipendenze Python

## File

```text
pyproject.toml
```

`sherpa-onnx` è già una dipendenza di VoiceMem.

Quindi Nemotron non richiede una nuova dipendenza ASR.

Mantenere:

```toml
"sherpa-onnx"
```

### FunASR e ModelScope

Non rimuovere immediatamente:

```toml
"funasr"
"modelscope"
```

solo come effetto collaterale della migrazione ASR.

Prima verificare se vogliamo usare `emotion2vec` tramite FunASR nel fork.

Decisione consigliata:

```text
ASR migration:
    non dipende da FunASR

emotion migration:
    decisione separata
```

Se successivamente emotion2vec viene adottato tramite `AutoModel`, FunASR/ModelScope rimangono necessari.

Se emotion2vec non viene adottato e nessun altro modulo usa FunASR/ModelScope, allora potranno essere rimossi.

---

# 9. Fase 5 — Modificare il sistema lingua da EN/ZH a EN/IT

## File

```text
voicemem/lang.py
```

Attualmente:

```python
SUPPORTED = ("en", "zh")
DEFAULT = "en"
```

Target:

```python
SUPPORTED = ("it", "en")
DEFAULT = "it"
```

È preferibile non introdurre una nuova serie di helper binari tipo:

```python
is_it()
```

al posto di:

```python
is_zh()
```

Meglio rendere il modulo realmente language-neutral.

---

## 9.1 Struttura proposta di `lang.py`

```python
SUPPORTED = ("it", "en")
DEFAULT = "it"

LANGUAGE_NAMES = {
    "it": "Italian",
    "en": "English",
}


def language_name(lang: str | None = None) -> str:
    lang = lang or memory_language()
    return LANGUAGE_NAMES[lang]


def label_rule() -> str:
    lang = language_name()

    return (
        f"Write every label in {lang}, whatever language the speaker used. "
        f"Do not mix in any other language."
    )
```

Rimuovere:

```python
is_zh()
```

e sostituire i call site con confronti generici:

```python
memory_language()
```

oppure mapping per lingua.

---

# 10. Fase 6 — Display delle emozioni EN/IT

Attualmente VoiceMem usa internamente otto emozioni canoniche rappresentate con stringhe cinesi.

Questa è una questione distinta dal supporto linguistico.

## Scelta consigliata per la prima implementazione

**Non cambiare ancora gli ID canonici interni.**

Motivo:

- sono usati da routing;
- anchor matching;
- quote;
- clustering;
- right brain;
- normalizzazione;
- confronti stringa;
- eventuali dati già persistiti.

Il cinese viene eliminato come **lingua supportata e lingua visibile**, non è necessario rinominare subito tutte le chiavi interne.

Aggiungere invece due mapping di display:

```python
_EMOTION_DISPLAY = {
    "en": {
        "焦虑": "anxious",
        "悲伤": "sad",
        "委屈": "wronged",
        "孤独": "lonely",
        "纠结": "conflicted",
        "平静": "calm",
        "开心": "happy",
        "疲惫": "tired",
    },
    "it": {
        "焦虑": "ansioso",
        "悲伤": "triste",
        "委屈": "ferito",
        "孤独": "solo",
        "纠结": "combattuto",
        "平静": "calmo",
        "开心": "felice",
        "疲惫": "stanco",
    },
}
```

e:

```python
def display_emotion(canonical: str) -> str:
    return _EMOTION_DISPLAY.get(
        memory_language(),
        _EMOTION_DISPLAY["it"],
    ).get(canonical, canonical)
```

### Fase successiva facoltativa

Se vogliamo un codebase senza nessuna stringa cinese anche internamente, convertire gli enum canonici in ID neutrali:

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

Questo è un refactor separato e non deve bloccare il port EN/IT.

---

# 11. Fase 7 — `anchor_router.py`

## File

```text
voicemem/rightbrain/anchor_router.py
```

Attualmente contiene:

```text
_EMOTION_KEYWORDS
_EMOTION_KEYWORDS_EN
```

Aggiungere:

```python
_EMOTION_KEYWORDS_IT
```

con termini come:

```text
ansioso
ansia
preoccupato
nervoso
agitato
stressato

triste
abbattuto
giù
dispiaciuto
deluso

arrabbiato
furioso
irritato
frustrato

solo
solitario
isolato

combattuto
indeciso
confuso
incerto

calmo
tranquillo
sereno
rilassato

felice
contento
entusiasta
allegro
soddisfatto
sollevato

stanco
esausto
sfinito
assonnato
```

Il normalizzatore deve riconoscere sia inglese sia italiano.

Non tradurre gli slot interni durante questa fase.

---

# 12. Fase 8 — Core language configuration

## File

```text
voicemem/core.py
```

Aggiornare documentazione/commenti:

```text
memory_language="en" o "it"
```

Default effettivo:

```text
it
```

La semantica deve rimanere:

> la lingua appartiene alla Memory Space, non al singolo utterance.

Questo è particolarmente importante per il nuovo ASR, perché la stessa proprietà determina anche:

```python
NemotronStreamingASR(language=...)
```

---

# 13. Fase 9 — Web backend

## File

```text
web/run.py
```

Il file contiene molte assunzioni binarie `zh/en`.

Devono diventare `it/en`.

---

## 13.1 `_LANG_NOTE`

Target:

```python
_LANG_NOTE = {
    "it": "Rispondi sempre in italiano, anche se l'utente usa un'altra lingua.",
    "en": "Always reply in English, even if the user uses another language.",
}
```

---

## 13.2 `space_language()`

Attualmente qualunque valore non cinese finisce in inglese.

Sostituire con validazione esplicita:

```python
def space_language(name: str) -> str:
    ...
    lang = str(v).strip().lower()

    if lang not in ("it", "en"):
        return "it"

    return lang
```

Mai usare logica del tipo:

```python
"en" if ... else "it"
```

quando il valore proviene da disco o API: gli errori di configurazione vanno intercettati.

---

## 13.3 Creazione Memory Space

Target:

```python
lang = str(language or ARGS.lang).strip().lower()

if lang not in ("it", "en"):
    raise ValueError(...)
```

Scrivere quindi:

```json
{
  "space": {
    "language": "it"
  }
}
```

oppure:

```json
{
  "space": {
    "language": "en"
  }
}
```

---

## 13.4 `UI_LANG`

Target:

```python
UI_LANG = ARGS.lang
SPACE_LANG = "it"
```

Default CLI:

```text
--lang it
```

---

## 13.5 Eliminare `is_zh()`

Esempio attuale:

```python
from voicemem.lang import is_zh
...
"zh" if is_zh() else "en"
```

Target:

```python
from voicemem.lang import memory_language
...
memory_language()
```

---

# 14. Fase 10 — Web UI

## File

```text
web/voicemem.html
```

Attualmente:

```javascript
const I18N = {
    zh: {...},
    en: {...}
}
```

Target:

```javascript
const I18N = {
    it: {...},
    en: {...}
}
```

---

## 14.1 Default UI

Attualmente il frontend parte in cinese.

Target:

```javascript
let LANG = localStorage.getItem('vm-lang') || 'it';
```

Fallback:

```javascript
const s =
    (I18N[LANG] && I18N[LANG][key]) ||
    I18N.it[key] ||
    key;
```

---

## 14.2 HTML lang

Target:

```javascript
document.documentElement.lang =
    LANG === 'en' ? 'en' : 'it';
```

---

## 14.3 Selettore lingua

Supportare esclusivamente:

```html
<option value="it">Italiano</option>
<option value="en">English</option>
```

Eliminare l'opzione cinese.

---

## 14.4 Slot UI

Mantenere gli ID interni:

```text
work
health
relationships
finance
goals
daily_life
knowledge
emotion
personality
preference
```

Aggiungere descrizioni italiane:

```javascript
it: {
    work: 'Lavoro · carriera / progetti / colleghi / riunioni',
    health: 'Salute · attività / alimentazione / sonno / corpo',
    relationships: 'Relazioni · famiglia / amici / partner / socialità',
    finance: 'Finanze · entrate / spese / affitto / investimenti',
    goals: 'Obiettivi · piani / futuro / crescita personale',
    daily_life: 'Vita quotidiana · abitudini / spostamenti / stile di vita',
    knowledge: 'Conoscenza · studio / lettura / competenze',
    emotion: 'Emozioni · stato attuale e andamento nel tempo',
    personality: 'Personalità · tendenze stabili e strategie di risposta',
    preference: 'Preferenze · gusti e stile di comunicazione',
}
```

---

# 15. Fase 11 — Prompt e memory extraction

La migrazione EN/IT non termina nell'ASR.

Devono essere ripuliti tutti i prompt che assumono cinese/inglese.

File principali:

```text
voicemem/reply.py
voicemem/leftbrain/extract_facts_openai.py
voicemem/leftbrain/mem0_additive_prompt_build.py
voicemem/leftbrain/cognitive_graph/query_slot_classifier.py
voicemem/rightbrain/*
web/run.py
```

---

## 15.1 `extract_facts_openai.py`

La regola attuale contiene esempi espliciti Chinese/English.

Target concettuale:

```text
Memory Space language is fixed.

If memory_language == "it":
    all free-text memories must be Italian.

If memory_language == "en":
    all free-text memories must be English.

Do not decide the storage language from the current utterance.
```

Questo è preferibile rispetto al comportamento:

```text
use_input_language=True
```

perché la lingua della Space deve essere la sorgente di verità.

---

## 15.2 Junk filter italiano

Aggiungere pattern italiani per evitare di memorizzare richieste transitorie o frasi dell'assistente.

Esempi:

```text
l'assistente ha suggerito
l'assistente ha consigliato
l'assistente ha risposto

ha chiesto un consiglio
ha chiesto suggerimenti
chiede una raccomandazione
vuole sapere

mi senti
riesci a sentirmi
ci sei
funziona
mi senti bene
```

Non fare un filtro troppo aggressivo basato su singole parole come:

```text
consiglio
vuole
chiede
```

perché possono comparire in fatti reali.

---

# 16. Fase 12 — Trigger semantici italiani

`web/run.py` contiene trigger lessicali per funzionalità speciali.

Non basta cambiare la UI.

Cercare almeno:

```text
sound/music triggers
temporal expressions
weekdays
locations
ordinals
playback requests
```

Aggiungere equivalenti italiani.

Esempi:

```text
ieri
l'altro ieri
avantieri
oggi
domani
settimana scorsa
mese scorso

lunedì
martedì
mercoledì
giovedì
venerdì
sabato
domenica

primo
prima
secondo
seconda
ultimo
ultima
penultimo
penultima

fammi sentire
riproduci
riascolta
metti la registrazione
che suono era
che musica era
```

Gestire sia apostrofo ASCII sia tipografico quando rilevante:

```text
l'altro
l’altro
un'ora
un’ora
```

---

# 17. Fase 13 — Ripulire il cinese dal runtime

Dopo le modifiche eseguire:

```bash
rg -n --pcre2 '\p{Han}' voicemem web examples tests
```

Classificare ogni match in:

1. **runtime string** → deve essere eliminata o tradotta;
2. **prompt/example** → deve essere eliminato o convertito EN/IT;
3. **UI** → deve essere eliminata;
4. **test fixture** → deve essere eliminata/sostituita;
5. **commento/docstring** → può essere tradotto in una fase successiva;
6. **canonical internal ID** → può rimanere temporaneamente solo se scelto esplicitamente.

Per verificare il supporto linguistico non basta cercare `"zh"`.

Cercare anche:

```bash
rg -n 'zh|Chinese|中文|chinese|zh-CN|zh_cn' .
```

---

# 18. Fase 14 — Test unitari ASR

Creare:

```text
tests/test_nemotron_asr.py
```

I test principali devono coprire il contratto del provider.

---

## 18.1 Test lingua italiana

Input WAV noto:

```text
tests/assets/asr_it.wav
```

Frase consigliata:

```text
Oggi devo ricordarmi di comprare il malto Pilsner.
```

Verificare che il final contenga almeno le parole chiave:

```text
oggi
comprare
malto
pilsner
```

Non testare l'uguaglianza byte-per-byte della trascrizione: punteggiatura e casing possono variare tra versioni del modello.

---

## 18.2 Test lingua inglese

Input:

```text
tests/assets/asr_en.wav
```

Esempio:

```text
Tomorrow I need to buy coffee and call Marco.
```

---

## 18.3 Test cumulative partial

Spezzare il WAV in chunk e chiamare:

```python
asr.feed(chunk)
```

Verificare:

- nessuna eccezione;
- ritorno sempre `str`;
- il testo non venga duplicato artificialmente;
- il final sia coerente con i partial.

---

## 18.4 Test `flush()`

Verificare che una frase che termina prima di riempire completamente il contesto venga finalizzata correttamente.

Questo test è importante perché VoiceMem usa `flush()` all'EOU.

---

## 18.5 Test `reset()`

Sequenza:

```text
utterance A
flush
reset
utterance B
flush
```

Il testo B non deve contenere testo di A.

---

## 18.6 Test lingua per stream

Creare due istanze:

```python
NemotronStreamingASR(..., language="it")
NemotronStreamingASR(..., language="en")
```

e verificare che entrambe funzionino sullo stesso modello.

---

# 19. Fase 15 — Integration test con `VoiceStream`

Creare un test che usa direttamente:

```python
vm.stream(...)
```

e fornisce PCM16 a chunk, come fa il frontend.

Pipeline da verificare:

```text
PCM
 ↓
resample
 ↓
Nemotron feed()
 ↓
partial
 ↓
speculative retrieval
 ↓
VAD endpoint
 ↓
Nemotron flush()
 ↓
turn_over
```

Testare almeno:

```text
Italian Space + Italian audio
English Space + English audio
```

---

# 20. Fase 16 — Test Memory Space

Verificare:

### Space italiana

```python
VoiceMem(
    space="italiano",
    memory_language="it",
)
```

JSON:

```json
{
  "space": {
    "language": "it"
  }
}
```

ASR:

```text
language=it
```

Memorie:

```text
italiano
```

Reply:

```text
italiano
```

---

### Space inglese

```python
VoiceMem(
    space="english",
    memory_language="en",
)
```

ASR:

```text
language=en
```

Memorie:

```text
English
```

Reply:

```text
English
```

---

# 21. Fase 17 — Test del cambio Space nella Web UI

Scenario:

```text
Space A = it
Space B = en
```

1. selezionare Space A;
2. iniziare una nuova sessione ASR;
3. verificare `language=it`;
4. chiudere/cambiare Space;
5. selezionare Space B;
6. verificare che venga costruito/reset un ASR con `language=en`.

### Attenzione

Non cambiare semplicemente la variabile `SPACE_LANG` lasciando vivo uno stream Nemotron creato con la vecchia lingua.

Al cambio Space:

- terminare lo stream corrente;
- creare/reset l'istanza VoiceMem corretta;
- assicurarsi che il prossimo `create_stream()` riceva la nuova lingua.

---

# 22. Fase 18 — Warmup

`VoiceMem.warmup()` in modalità multimodale carica le capability necessarie.

Verificare che:

```text
asr
```

carichi Nemotron una sola volta.

Il recognizer è pesante; non va ricreato a ogni utterance.

Da ricreare a ogni turno è soltanto:

```text
OnlineStream
```

non:

```text
OnlineRecognizer
```

Target:

```text
startup:
    load OnlineRecognizer ~ una volta

turn 1:
    create stream
    decode
    reset -> new stream

turn 2:
    create stream
    decode
```

---

# 23. Fase 19 — Performance test su Mac

Misurare almeno:

```text
Mac model
CPU
RAM
macOS
Python
sherpa-onnx version
chunk model
```

Metriche:

```text
recognizer load time
RAM after load
CPU during speech
RTF
time to first partial
finalization latency after VAD
```

Corpus minimo:

```text
10 utterance italiane
10 utterance inglesi
```

con:

```text
frasi brevi
frasi lunghe
nomi propri
numeri
date
termini inglesi dentro frase italiana
```

Il modello `560ms` è la baseline.

Solo se la latenza percepita è troppo alta confrontare:

```text
160ms
560ms
```

Non partire dal modello più piccolo solo perché teoricamente è più realtime: abbiamo già verificato che 560 ms funziona bene e l'accuratezza è prioritaria.

---

# 24. Fase 20 — Error handling

Il provider deve fallire subito se manca un file.

Nel costruttore:

```python
required = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
]
```

Per ogni file:

```python
if not path.exists():
    raise FileNotFoundError(...)
```

Messaggio utile:

```text
Nemotron ASR model not found.
Run: bash scripts/download_models.sh
Expected: models/asr/...
```

Non lasciare che sherpa fallisca più tardi con un errore ONNX poco leggibile.

---

# 25. Fase 21 — Configurazione consigliata

Non serve esporre decine di parametri.

Configurazione minima:

```text
memory_language
ASR model path
ASR num_threads
```

Opzionale via env:

```text
VOICEMEM_ASR_THREADS
```

Esempio:

```python
num_threads = int(
    os.environ.get("VOICEMEM_ASR_THREADS", "4")
)
```

Non aggiungere per ora:

```text
provider selector
chunk selector runtime
automatic language detection
beam search tuning
hotword UI
```

Possono essere aggiunti solo se emerge un requisito reale.

---

# 26. Fase 22 — Documentazione utente

Aggiornare:

```text
README.md
examples/
scripts/download_models.sh
```

Quick start proposto:

```bash
git clone <fork>
cd <fork>

pip install -e .
bash scripts/download_models.sh

python web/run.py
```

Lingue:

```text
Italiano
English
```

Rimuovere dalla documentazione:

```text
Chinese
zh
paraformer-zh-streaming
bilingual-zh-en Zipformer
```

---

# 27. Fase 23 — Emotion recognition

La migrazione Nemotron non deve tentare di estrarre emozioni dall'ASR.

Architettura corretta:

```text
                    ┌─> Nemotron -> transcript
utterance audio ----┤
                    └─> emotion detector -> emotion
```

Nemotron risponde a:

```text
cosa è stato detto?
```

Il modulo emozionale risponde a:

```text
come è stato detto?
```

Il test fatto con `emotion2vec_plus_base` dimostra che può essere usato come candidato, ma la sua integrazione deve rimanere un task separato dalla migrazione ASR.

Non modificare il contratto ASR per aggiungere:

```text
emotion
confidence
valence
arousal
```

L'ASR deve continuare a restituire testo.

---

# 28. Lista file da modificare

## P0 — necessari per far funzionare ASR EN/IT

```text
voicemem/utils/audio/asr.py
voicemem/utils/defaults.py
voicemem/lang.py
scripts/download_models.sh
voicemem/core.py
```

## P0 — necessari per rendere davvero il progetto EN/IT

```text
web/run.py
web/voicemem.html
voicemem/reply.py
voicemem/leftbrain/extract_facts_openai.py
voicemem/rightbrain/anchor_router.py
```

## P1 — audit dei prompt e del comportamento linguistico

```text
voicemem/leftbrain/mem0_additive_prompt_build.py
voicemem/leftbrain/cognitive_graph/query_slot_classifier.py
voicemem/rightbrain/*
voicemem/utils/audio/emotion/*
examples/*
tests/*
README.md
```

## P2 — pulizia sorgente

```text
commenti cinesi
docstring cinesi
evaluation/
finetune/
documentazione upstream non più rilevante
```

---

# 29. Ordine di implementazione consigliato

## Step 1

Aggiungere:

```text
NemotronStreamingASR
```

senza toccare ancora la UI.

Test:

```text
WAV italiano
WAV inglese
feed/flush/reset
```

---

## Step 2

Cambiare:

```text
default_utils.asr()
```

per usare sempre Nemotron.

Test:

```text
VoiceStream
```

con audio preregistrato.

---

## Step 3

Modificare:

```text
scripts/download_models.sh
```

e verificare installazione da repository pulito.

---

## Step 4

Cambiare:

```text
SUPPORTED = ("it", "en")
DEFAULT = "it"
```

in `lang.py`.

---

## Step 5

Collegare la lingua della Memory Space al Nemotron stream.

Verificare esplicitamente:

```text
Space IT -> set_option("language", "it")
Space EN -> set_option("language", "en")
```

---

## Step 6

Convertire il backend web da:

```text
zh/en
```

a:

```text
it/en
```

---

## Step 7

Convertire `web/voicemem.html`:

```text
I18N.zh -> I18N.it
default zh -> it
```

---

## Step 8

Aggiornare prompt, junk filter, trigger e keyword emozionali italiani.

---

## Step 9

Eseguire audit:

```bash
rg -n 'zh|Chinese|中文|zh-CN|zh_cn' .
rg -n --pcre2 '\p{Han}' voicemem web examples tests
```

---

## Step 10

Eseguire test funzionali completi:

```text
Italian conversation
English conversation
IT -> EN space switch
EN -> IT space switch
memory ingest
memory retrieval
reply language
ASR partial
ASR final
emotion pipeline
web UI
```

---

# 30. Criteri di accettazione

La migrazione ASR può essere considerata completata quando:

- [ ] Paraformer non viene più caricato.
- [ ] Il vecchio Zipformer zh/en non viene più caricato.
- [ ] Nemotron è il solo ASR interno.
- [ ] Un solo checkpoint Nemotron gestisce EN e IT.
- [ ] `feed()` restituisce partial cumulativi.
- [ ] `flush()` restituisce il final completo.
- [ ] `reset()` elimina completamente lo stato del turno precedente.
- [ ] Una Space italiana forza `language="it"`.
- [ ] Una Space inglese forza `language="en"`.
- [ ] Il recognizer non viene ricaricato a ogni utterance.
- [ ] Il progetto non usa language autodetection nel percorso standard.
- [ ] `SUPPORTED` contiene solo `it` ed `en`.
- [ ] Il frontend offre solo Italiano ed English.
- [ ] Una nuova Space nasce di default in italiano.
- [ ] Le memorie di una Space IT vengono scritte in italiano.
- [ ] Le memorie di una Space EN vengono scritte in inglese.
- [ ] Le risposte seguono la lingua della Space.
- [ ] I trigger italiani principali funzionano.
- [ ] Le keyword emozionali italiane vengono normalizzate.
- [ ] Non rimangono prompt runtime dipendenti dal cinese.
- [ ] I test EN/IT passano su macOS.
- [ ] Un'installazione pulita scarica il modello corretto con un solo comando.

---

# 31. Cose da NON fare durante questa migrazione

Non:

- riscrivere `VoiceStream`;
- introdurre Whisper come fallback;
- mantenere tre backend ASR "per sicurezza";
- usare autodetect della lingua quando la Space la conosce già;
- ricreare il recognizer Nemotron a ogni frase;
- fondere emotion recognition dentro il provider ASR;
- tradurre subito tutti gli ID interni degli slot;
- cambiare embedding senza benchmark;
- cambiare VAD solo perché cambia l'ASR;
- cambiare contemporaneamente ASR, emotion model e memory schema nello stesso commit.

La migrazione deve rimanere isolabile e reversibile durante lo sviluppo.

---

# 32. Strategia commit consigliata

```text
1. feat(asr): add Nemotron multilingual streaming provider
2. feat(asr): make Nemotron the default and remove legacy ASR selection
3. chore(models): download Nemotron 3.5 streaming model
4. feat(lang): replace zh support with it and default to Italian
5. feat(web): migrate backend language handling to it/en
6. feat(ui): replace Chinese locale with Italian locale
7. feat(memory): enforce memory-space language in prompts
8. feat(rightbrain): add Italian emotion normalization
9. test(asr): add Italian and English streaming fixtures
10. test(lang): add IT/EN memory-space integration tests
11. docs: update README and installation
12. chore: remove remaining Chinese runtime strings
```

Questo ordine permette di testare ogni modifica separatamente.

---

# 33. Decisioni aperte dopo il port ASR

Da affrontare solo dopo che Nemotron EN/IT è stabile:

### Emotion engine

Confrontare:

```text
emotion2vec_plus_base
vs
pipeline emozionale VoiceMem esistente
vs
fusione audio + testo
```

### Internal emotion IDs

Decidere se sostituire definitivamente gli attuali canonical ID con enum neutrali inglesi.

### TTS

Scegliere voci IT/EN e comportamento per Memory Space.

### Prompt quality

Costruire un piccolo corpus italiano per:

```text
fact extraction
junk rejection
temporal queries
slot classification
right-brain traits
```

### ASR chunk

Confrontare soltanto se necessario:

```text
160 ms
560 ms
```

La baseline resta `560ms-int8`.

---

# 34. Riferimenti tecnici

VoiceMem upstream:

https://github.com/xzf-thu/VoiceMem

Nemotron 3.5 ASR Streaming in sherpa-onnx:

https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html

Modello scelto:

```text
sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
```

Repository sherpa-onnx:

https://github.com/k2-fsa/sherpa-onnx

---

# 35. Risultato target finale

```text
VoiceMem EN/IT
│
├── Language
│   ├── it   [default]
│   └── en
│
├── ASR
│   └── Nemotron 3.5 Streaming 0.6B int8
│       ├── language=it
│       └── language=en
│
├── VAD
│   └── invariato
│
├── Audio perception
│   ├── voiceprint
│   ├── scene
│   └── emotion
│
├── Memory
│   ├── Space IT -> testi italiani
│   └── Space EN -> testi inglesi
│
├── Reply
│   ├── Space IT -> risposta italiana
│   └── Space EN -> risposta inglese
│
└── Web UI
    ├── Italiano
    └── English
```

Il punto chiave è che **la lingua della Memory Space diventa la singola sorgente di verità** per ASR, memoria e risposta. Nemotron permette di farlo con un solo checkpoint e senza introdurre un secondo stack di riconoscimento.
