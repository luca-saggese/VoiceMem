<a id="italian"></a>

<p align="center">
  <img src="assets/Voicemem_logo.webp" alt="VoiceMem Logo" width="100%">
</p>

<p align="center">
  <strong>Italiano</strong> | <a href="#english">English</a>
</p>

<p align="center">
  <a href="https://xzf-thu.github.io/VoiceMem/">Pagina del progetto 🌐</a> /
  <a href="https://arxiv.org/pdf/2608.26005">Report tecnico 📖</a> /
  <a href="https://huggingface.co/zhifeixie/VoiceMem_Default_Models_Env">VoiceMem Utils 🤗</a> /
  <a href="https://huggingface.co/zhifeixie/VoiceMem_MF_Qwen3_6_35B_A3B_Qlora">VoiceMem Model Families 🤗</a> /
  <a href="https://huggingface.co/datasets/zhifeixie/VoiceMem-ChatMem400k">ChatMem-400K 🤗</a>
</p>

<p align="center">
  <a href="wechat.jpg">
    <img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat">
  </a>
  <a href="https://x.com/XieZhifei14110">
    <img src="https://img.shields.io/badge/X-@XieZhifei14110-black?logo=x&logoColor=white" alt="X">
  </a>
  <a href="https://xzf-thu.github.io">
    <img src="https://img.shields.io/badge/Personal-Contact-blue" alt="Personal Contact">
  </a>
</p>

<div align="center">
  <a href="https://xzf-thu.github.io/VoiceMem/">
    <img src="assets/huggingface_paper_gold_day.svg"/>
  </a>
</div>
<p align="center">
  <img src="assets/wechat.jpg" alt="Gruppo WeChat di VoiceMem" width="60%">
</p>

---

Presentiamo **VoiceMem**, il componente finale per i modelli vocali: un'anima che permette loro di conoscerti davvero sempre meglio. VoiceMem si basa su un'architettura **a doppio cervello in streaming** e offre un servizio di memoria **preciso, emotivo, consapevole della personalità, a bassa latenza e a costi minimi**. Questo repository resterà **completamente open source per sempre**.

Una panoramica rapida di VoiceMem:

* **Cervello sinistro:** gestisce direttamente le informazioni e mantiene le prestazioni complete di Mem0 con un limite di Top-3 memorie.
* **Cervello destro:** gestisce l'intelligenza emotiva tramite l'attribuzione delle emozioni a breve e lungo termine, con nodi tra entità e manutenzione congiunta delle informazioni del cervello sinistro.
* **Bassa latenza:** comprime le informazioni, usa l'archiviazione gerarchica e il recupero in streaming con prefetch speculativo da 0 a 300 ms, aggiungendo una latenza quasi nulla.
* **Semplice e pratico:** ogni query usa circa 300 token. L'architettura è completamente disaccoppiata e ogni componente, incluso il motore di memoria sottostante, può essere sostituito.

<p align="center">
  <img src="assets/teaser.webp" alt="Panoramica di VoiceMem" width="100%">
</p>

## 🔥 Novità

* **27/08/2026 · v0.0.1** — Pubblicati la prima versione di **VoiceMem** e il **Technical Report**.

## 🎬 Demo

> **Nota:** disattiva l'audio prima di avviare il video.

https://github.com/user-attachments/assets/0d919f8c-e9ba-4fdb-8078-b049e4b99a28


## 📚 Indice
* [🚀 Avvio rapido](#-avvio-rapido)
* [🧠 Architettura a doppio cervello in streaming di VoiceMem](#-voicemem-sistema-di-memoria-con-architettura-a-doppio-cervello-in-streaming)
* [🤖 Famiglie di modelli VoiceMem](#-famiglie-di-modelli-voicemem)
* [🔌 Personalizza il tuo agente vocale con VoiceMem](#-personalizza-il-tuo-agente-vocale-con-voicemem)
* [🛠️ Fine-tuning dei modelli](#️-fine-tuning-dei-modelli)
* [📊 Valutazione](#-valutazione)
* [Ringraziamenti](#ringraziamenti)
* [Licenza](#licenza)

## 🚀 Avvio rapido

### Installazione

```bash
git clone https://github.com/lang-jiaqi/Voicemem_open.git
cd Voicemem_open

# Installa il sistema di memoria (include ASR / identificazione del parlante / scena / emozioni / embedding locali)
pip install voicemem

# Opzionale: usa il nostro modello Qwen per le risposte, sottoposto a fine-tuning
pip install "voicemem[slm]"
```

### Download dei modelli richiesti

```bash
pip install -U huggingface_hub

hf download zhifeixie/VoiceMem_Default_Models_Env --local-dir ./models
```

### Utilizzo di base <a id="interfaces"></a>

#### Esecuzione come motore di memoria offline

```python
from voicemem import VoiceMem

vm = VoiceMem(
    mode="normal",
    openai_key="api_xxx",
    top_k=5,
)

# I modelli locali vengono caricati in modo lazy: riscaldali per evitare attese alla prima chiamata
vm.warmup()

# Memorizza un file audio.
# VoiceMem esegue internamente ASR / identificazione del parlante / scena / emozioni / estrazione degli embedding.
print("inizio ingest")
vm.ingest(audio="assets/input.wav")  # Sono vegetariano e allergico alla frutta secca.
print("fine ingest")

# La scrittura è lenta perché estrae fatti, assegna tag e costruisce il grafo.
# La lettura è una ricerca vettoriale pura, indipendente dal costo di scrittura.
print("inizio ricerca")
result = vm.search("Quali sono le mie restrizioni alimentari?")
print("fine ricerca")

print(result.result_leftbrain, result.result_rightbrain)


# Memorizza direttamente testo fattuale nel cervello sinistro (senza informazioni emotive).
vm = VoiceMem(
    mode="leftbrain_only",
    openai_key="api_xxx",
    top_k=5,
)

vm.ingest("Sono vegetariano e allergico alla frutta secca.")

result = vm.search("Quali sono le mie restrizioni alimentari?")
```

#### Esecuzione di VoiceMem in modalità streaming

L'interfaccia streaming di VoiceMem può essere considerata un'interfaccia VAD che elabora continuamente l'audio.

L'esempio seguente memorizza prima esplicitamente un fatto, poi invia una traccia audio contenente una **domanda** per mostrare come la memoria venga recuperata prima che il parlante finisca di parlare; infine esegue la normale decisione di ingest.

```python
import asyncio
import os
from pprint import pprint

import numpy as np
import soundfile as sf

from voicemem import VoiceMem

# Riutilizza il vm precedente; ne crea uno qui per eseguire il blocco in autonomia
vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

# I modelli locali vengono caricati in modo lazy: riscaldali per evitare attese al primo blocco audio
vm.warmup()

# Memorizza prima un fatto, così la domanda seguente avrà qualcosa da trovare
vm.ingest("Sono vegetariano e allergico alla frutta secca.")

SPEC_MIN_CHARS = 6          
searching = False


def on_partial(text):
    """Mostra la trascrizione parziale; se è abbastanza lunga, la ricerca è già iniziata."""
    global searching
    print(f"\r[partial] {text}", end="", flush=True)
    if not searching and len(text) >= SPEC_MIN_CHARS:
        searching = True
        print("\n[inizio ricerca] il parlante non ha ancora finito, ma il recupero è già in corso", flush=True)


async def main():
    # Questa traccia audio contiene la domanda: "Quali sono le mie restrizioni alimentari?"
    audio, sr = sf.read("assets/question.wav", dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stream = vm.stream(src_rate=sr, vad_threshold=0.5, on_partial=on_partial)
    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        if st.state != "turn_over":                
            continue

        # Il VAD conferma la fine del turno. La memoria è stata recuperata durante il parlato: basta leggerla
        print("[fine ricerca]")
        print("trascrizione  ", st.transcript)
        print("cervello sinistro  ", st.result_leftbrain)
        print("cervello destro  ", st.result_rightbrain)
        pprint({k: getattr(st, k) for k in
                ["speaker_id", "speaker_voiceprint", "emotion",
                 "entity", "schema", "text_embedding"]})

        # Ogni turno esegue una decisione di ingest
        print("[ingest] l'LLM sta decidendo se vale la pena memorizzare questa frase...", flush=True)
        res = vm.ingest(st.transcript)
        print(f"[ingest] estratti {res['facts_count']} fatti -> {res['memory_ids']}")
       


asyncio.run(main())
```

### Demo interattiva con VoiceMem

Il codice della demo si trova nel repository (il pacchetto installato con pip contiene solo la libreria); assicurati di aver clonato il repository e di trovarti nella sua directory principale.

```bash
python web/run.py
```

Poi apri:

```text
http://localhost:8787
```

## 🧠 VoiceMem: sistema di memoria con architettura a doppio cervello in streaming

**VoiceMem** è un sistema di memoria progettato per agenti vocali in tempo reale.

Invece di archiviare ogni tipo di memoria in un unico database di ricerca, VoiceMem separa la memoria in due parti complementari:

<p align="center">
  <img src="docs/images/fig-architecture.webp" alt="Architettura di VoiceMem" width="80%">
</p>

* Il **cervello sinistro** organizza la memoria fattuale tramite schemi ed entità, per recuperare le informazioni con maggiore precisione.
* Il **cervello destro** gestisce personalità, emozioni e relazioni tramite nodi indipendenti e nodi tra entità.

<p align="center">
  <img src="docs/images/stages.webp" alt="Pipeline di elaborazione di VoiceMem" width="90%">
</p>

L'intera pipeline è **in streaming**.

Mentre l'utente sta ancora parlando, VoiceMem segmenta continuamente l'audio, trascrive il parlato, estrae le memorie utili e scrive le informazioni strutturate nel grafo della memoria.

Durante una query, VoiceMem **esegue prima il routing, poi il ranking e infine inserisce nel contesto del modello solo le memorie Top-K**, mantenendo il contesto compatto e conservando le informazioni più rilevanti.

### Caratteristiche principali

* 🎯 **Preciso** — Raggiunge **91,2% su LoCoMo**, rispetto al **61,68% di Mem0**, usando solo **Top-5** memorie.
* ❤️ **Emotivo e personale** — Ricorda non solo **cosa ha detto l'utente**, ma anche **chi è e come si sente**. Raggiunge **69,44% su PersonaMem**.
* 🎧 **Multimodale** — Ricorda **parlato, parlanti, eventi sonori, conversazioni con più interlocutori e musica** da audio del mondo reale.
* ⚡ **Veloce** — Risponde in **134 ms**, rispetto ai **1.440 ms di Mem0**, con recupero streaming durante il turno vocale.
* 💰 **Basso consumo di token** — Usa solo **430 token di memoria**, rispetto ai **6.956 di Mem0** e ai **1.899 di EverMemOS**.



## 🤖 Famiglie di modelli VoiceMem

Costruiamo **ChatMem-400K** tramite una pipeline di training OPD in tre fasi:

1. **Memory-world construction**
2. **SLM-validated online on-policy distillation（OPD）**
3. **Human refinement**

La stessa pipeline, dopo la revisione umana, produce **ChatMem-Bench**, che valuta se un modello vocale sia in grado di sviluppare nel tempo una comprensione dell'utente.

La famiglia di modelli open source di VoiceMem include **Qwen2.5-Omni, Qwen3-Omni e Step-Audio2-Mini**. Questi modelli possono ricevere e comprendere le informazioni di memoria fornite da VoiceMem durante le conversazioni.

<p align="center">
  <img src="docs/images/fig-opd.webp" alt="Pipeline OPD di VoiceMem" width="90%">
</p>

## 🔌 Personalizza il tuo agente vocale con VoiceMem

Puoi integrare VoiceMem con il tuo modello vocale per costruire un agente vocale in tempo reale dotato di memoria a lungo termine.

Il flusso di base è:

**microfono → VoiceMem ascolta e recupera in anticipo le memorie rilevanti → il tuo modello legge le memorie e genera una risposta**

```bash
export OPENAI_API_KEY=sk-...
# Usata solo per l'estrazione dei fatti durante la scrittura delle memorie.
# Il recupero delle memorie è interamente locale.

python examples/03_simple_agent_with_voicemem_memory.py
```

Per usare il tuo modello, sostituisci il passaggio di generazione: la parte relativa alla memoria resta invariata.

```python
def my_reply(text, memory_context):        # anche le funzioni sincrone funzionano, verranno automaticamente girate in un thread
    return my_model.generate(system=memory_context, user=text)

vm = VoiceMem(reply=my_reply)
```

## 🛠️ Fine-tuning dei modelli

VoiceMem fornisce l'intera pipeline di fine-tuning per addestrare un adapter della tua VoiceMem Model Family.

La configurazione di training predefinita corrisponde a quella usata per il `checkpoint-3318` pubblicato.

Eseguendo il comando seguente con le impostazioni predefinite puoi riprodurre lo stesso adapter:

```bash
pip install ms-swift==4.5.2 bitsandbytes

python finetune/train.py --data data/train.jsonl
```

Per il formato dei dati di training, i requisiti di memoria GPU e le istruzioni per usare un modello di base diverso, consulta **[finetune/README.md](finetune/README.md)**.

## 📊 Valutazione

La pipeline di valutazione è completamente open source e riproducibile.

<p align="center">
  <img src="assets/evaluation.webp" alt="Risultati della valutazione di VoiceMem" width="100%">
</p>

### Esecuzione della valutazione

Puoi avviare un benchmark con un solo comando:

```bash
export OPENAI_API_KEY=sk-...

# Inizia con il piccolo esempio incluso nel repository
# per verificare che l'ambiente sia configurato correttamente.
# 2 conversazioni, 5 domande.
python evaluation/run.py \
    --dataset locomo \
    --data evaluation/examples/locomo_sample.json

# Poi esegui il dataset completo.
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json
```

Risultato di esempio:

```text
LoCoMo: 10 conversations · 152 questions

Score: 139/152 = 91.4%

  multi_hop     88.2%
  temporal      85.7%
  single_hop    95.1%

Median retrieval latency: 12 ms
Median retrieved memory: 298 tokens
```

Prima di eseguire una valutazione completa, aggiungi `--inspect` per verificare come viene analizzato il dataset.

Questa modalità non chiama il modello e non genera costi API:

```bash
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json \
    --inspect
```

Durante la valutazione, il modello che genera le risposte riceve **solo le memorie recuperate**, non la cronologia originale della conversazione.

Se il modello ricevesse la conversazione completa, il benchmark valuterebbe la sua capacità di comprensione del testo invece di quella del sistema di memoria.

Per il protocollo completo di valutazione e le istruzioni per aggiungere un nuovo benchmark, consulta **[evaluation/README.md](evaluation/README.md)**. Aggiungere un benchmark richiede solo un file e due funzioni.

## Ringraziamenti

Ringraziamo i seguenti eccellenti progetti open source:

* [mem0](https://github.com/mem0ai/mem0) — Motore di memoria vettoriale
* [FunASR](https://github.com/modelscope/FunASR) — ASR streaming basato su `paraformer-zh-streaming`
* [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — Silero VAD, verifica del parlante 3D-Speaker e ASR streaming di riserva
* [intfloat/multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small) — Embedding locale e classificazione Slot

VoiceMem usa inoltre le API OpenAI per le funzionalità Chat, TTS e Realtime.

## Licenza

VoiceMem è distribuito come open source secondo la **Apache License 2.0**.

Per i dettagli, consulta [LICENSE](LICENSE).

<br>

---

<br>

<a id="english"></a>

<p align="center">
  <img src="assets/Voicemem_logo.webp" alt="VoiceMem Logo" width="100%">
</p>

<p align="center">
  <a href="#chinese">中文</a> | <strong>English</strong>
</p>

<p align="center">
  <a href="https://xzf-thu.github.io/VoiceMem/">Project Page 🌐</a> /
  <a href="https://arxiv.org/pdf/2608.26005">Technical Report 📖</a> /
  <a href="https://huggingface.co/zhifeixie/VoiceMem_Default_Models_Env">VoiceMem Utils 🤗</a> /
  <a href="https://huggingface.co/zhifeixie/VoiceMem_MF_Qwen3_6_35B_A3B_Qlora">VoiceMem Model Families 🤗</a> /
  <a href="https://huggingface.co/datasets/zhifeixie/VoiceMem-ChatMem400k">ChatMem-400K 🤗</a>
</p>

<p align="center">
  <a href="wechat.jpg">
    <img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat">
  </a>
  <a href="https://x.com/XieZhifei14110">
    <img src="https://img.shields.io/badge/X-@XieZhifei14110-black?logo=x&logoColor=white" alt="X">
  </a>
  <a href="https://xzf-thu.github.io">
    <img src="https://img.shields.io/badge/Personal-Contact-blue" alt="Personal Contact">
  </a>
</p>

<p align="center">
  <img src="assets/wechat.jpg" alt="VoiceMem WeChat Group" width="60%">
</p>

---

We introduce **VoiceMem**, adding the final component to voice models: a soul, so they truly come to understand you better over time. VoiceMem is built on a <strong>streaming dual-brain</strong> architecture and provides **accurate, emotional, personality-aware, low-latency, and lowest-cost memory services**. This repository will <strong>remain fully open source, permanently</strong>.

A quick overview of VoiceMem:

* **Left Brain:** Directly manages factual information and sustains Mem0's full performance under a Top-3 memory limit.
* **Right Brain:** Manages emotional intelligence through short-term and long-term emotional attribution, including cross-entity nodes and joint maintenance with Left Brain information.
* **Low Latency:** Uses information compression, hierarchical storage, and streaming retrieval with 0–300 ms speculative prefetching, adding almost no extra latency.
* **Simple and Practical:** Each query uses about 300 tokens. The architecture is fully decoupled, and every component, including the underlying memory engine, can be replaced.

<p align="center">
  <img src="assets/teaser.webp" alt="VoiceMem Overview" width="100%">
</p>

## 🔥 News

* **08/27/2026 · v0.0.1** — Released the first version of **VoiceMem** and our **Technical Report**.

## 🎬 Demo Video

> **Note:** Please unmute the video before playback.
https://github.com/user-attachments/assets/0d919f8c-e9ba-4fdb-8078-b049e4b99a28

## 📚 Overview

* [🚀 Quick Start](#-quick-start)
* [🧠 VoiceMem Dual-Brain Streaming Architecture](#-voicemem-memory-with-a-streaming-dual-brain-architecture)
* [🤖 VoiceMem Model Families](#-voicemem-model-families)
* [🔌 Customize Your Voice Agent with VoiceMem](#-customize-your-voice-agent-with-voicemem)
* [🛠️ Finetuning](#️-finetuning)
* [📊 Evaluation](#-evaluation)
* [Acknowledgements](#acknowledgements)
* [License](#license)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/lang-jiaqi/Voicemem_open.git
cd Voicemem_open

# Install the memory system (bundles ASR / speaker ID / scene / emotion / local embedding)
pip install voicemem

# Optional: run our fine-tuned Qwen reply model
pip install "voicemem[slm]"
```

### Required Model Download

```bash
pip install -U huggingface_hub

hf download zhifeixie/VoiceMem_Default_Models_Env --local-dir ./models
```

### Basic Usage <a id="interfaces-en"></a>

#### Run as an Offline Memory Engine

```python
from voicemem import VoiceMem

vm = VoiceMem(
    mode="normal",
    openai_key="api_xxx",
    top_k=5,
)

# Local models load lazily -- warm them up so the first call doesn't pay for it.
vm.warmup()

# Store an audio file.
# VoiceMem internally runs ASR / speaker ID / scene / emotion / embedding extraction.
print("ingest start")
vm.ingest(audio="assets/input.wav")  # I am vegetarian and allergic to nuts.
print("ingest done")

# Writing is slow because it extracts facts, tags them and builds the graph.
# Reading is a pure vector lookup -- independent of write cost.
print("search start")
result = vm.search("What are my dietary restrictions?")
print("search done")

print(result.result_leftbrain, result.result_rightbrain)


# Store Left Brain factual text directly (no emotional information).
vm = VoiceMem(
    mode="leftbrain_only",
    openai_key="api_xxx",
    top_k=5,
)

vm.ingest("I am vegetarian and allergic to nuts.")

result = vm.search("What are my dietary restrictions?")
```

#### Run VoiceMem in Streaming Mode

Think of VoiceMem's streaming interface as a VAD interface that continuously processes audio.

The example below stores one fact explicitly, then feeds a **question** as audio to show how the memory is already retrieved before the speaker finishes. It ends, as always, with the ingest decision.

```python
import asyncio
import os
from pprint import pprint

import numpy as np
import soundfile as sf

from voicemem import VoiceMem

# Reuses the vm above; building one here so the block runs standalone
vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

# Local models load lazily -- warm them up so the first audio chunk doesn't wait
vm.warmup()

# Store one fact first, so the question below has something to find
vm.ingest("I am vegetarian and allergic to nuts.")

SPEC_MIN_CHARS = 6
searching = False


def on_partial(text):
    """Partial transcripts as they arrive. Long enough = the search already started."""
    global searching
    print(f"\r[partial] {text}", end="", flush=True)
    if not searching and len(text) >= SPEC_MIN_CHARS:
        searching = True
        print("\n[search start] speaker isn't done yet, retrieval already running", flush=True)


async def main():
    # This audio is a question: "What are my dietary restrictions?"
    audio, sr = sf.read("assets/question.wav", dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stream = vm.stream(src_rate=sr, vad_threshold=0.5, on_partial=on_partial)
    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        if st.state != "turn_over":
            continue

        # VAD confirmed end of turn. Memory was fetched while the user spoke -- just read it
        print("[search end]")
        print("transcript  ", st.transcript)
        print("left brain  ", st.result_leftbrain)
        print("right brain ", st.result_rightbrain)
        pprint({k: getattr(st, k) for k in
                ["speaker_id", "speaker_voiceprint", "emotion",
                 "entity", "schema", "text_embedding"]})

        # Every turn runs the ingest decision
        print("[ingest] LLM deciding whether this is worth storing...", flush=True)
        res = vm.ingest(st.transcript)
        print(f"[ingest] extracted {res['facts_count']} facts -> {res['memory_ids']}")


asyncio.run(main())
```

### Interactive Demo with VoiceMem

The demo lives in the repo (the pip package ships the library only) — make sure you have cloned it and are in the repo root.

```bash
python web/run.py
```

Then open:

```text
http://localhost:8787
```

## 🧠 VoiceMem: Memory with a Streaming Dual-Brain Architecture

**VoiceMem** is a memory system built for real-time voice agents.

Instead of storing every type of memory in a single retrieval database, VoiceMem separates memory into two complementary parts:

<p align="center">
  <img src="docs/images/fig-architecture.webp" alt="VoiceMem Architecture" width="80%">
</p>

* **Left Brain** organizes factual memory using schemas and entities for more accurate retrieval.
* **Right Brain** manages personality, emotion, and relationships using independent and cross-entity memory nodes.

<p align="center">
  <img src="docs/images/stages.webp" alt="VoiceMem Processing Pipeline" width="90%">
</p>

The entire pipeline is **streaming**.

While the user is still speaking, VoiceMem continuously segments audio, transcribes speech, extracts useful memories, and writes structured information into the memory graph.

At query time, VoiceMem **routes first, ranks second, and injects only the Top-K memories into the model context**. This keeps the context small while preserving the most relevant information.

### Key Features

* 🎯 **Accurate** — Reaches **91.2% on LoCoMo**, compared with **61.68% for Mem0**, using only **Top-5** memories.
* ❤️ **Emotional & Personal** — Remembers not only **what the user said**, but also **who the user is and how they feel**. Reaches **69.44% on PersonaMem**.
* 🎧 **Multimodal** — Remembers **speech, speakers, sound events, multi-speaker conversations, and music** from real-world audio.
* ⚡ **Fast** — Responds in **134 ms**, compared with **1,440 ms for Mem0**, with streaming retrieval inside the voice turn.
* 💰 **Low Token Usage** — Uses only **430 memory tokens**, compared with **6,956 for Mem0** and **1,899 for EverMemOS**.

---

## 🤖 VoiceMem Model Families

We build **ChatMem-400K** through a three-stage OPD training pipeline:

1. **Memory-world construction**
2. **SLM-validated online on-policy distillation (OPD)**
3. **Human refinement**

After human editing, the same pipeline produces **ChatMem-Bench**, which evaluates whether a voice model can build a long-term understanding of the user over time.

The open-source VoiceMem model family includes **Qwen2.5-Omni, Qwen3-Omni, and Step-Audio2-Mini**. These models can receive and understand memory information provided by VoiceMem during conversations.

<p align="center">
  <img src="docs/images/fig-opd.webp" alt="VoiceMem OPD Pipeline" width="90%">
</p>

## 🔌 Customize Your Voice Agent with VoiceMem

You can integrate VoiceMem with your own voice model to build a real-time voice agent with long-term memory.

The basic flow is:

**microphone → VoiceMem listens and prefetches relevant memories → your model reads those memories and generates a response**

```bash
export OPENAI_API_KEY=sk-...
# Only used for fact extraction when writing memories.
# Memory retrieval runs entirely locally.

python examples/03_simple_agent_with_voicemem_memory.py
```

To use your own model, replace the generation step — the memory half stays exactly as is:

```python
def my_reply(text, memory_context):        # a sync function is fine, it runs off-thread
    return my_model.generate(system=memory_context, user=text)

vm = VoiceMem(reply=my_reply)
```

## 🛠️ Finetuning

VoiceMem provides the complete finetuning pipeline for training your own VoiceMem Model Family adapter.

The default training configuration matches the one used for the released `checkpoint-3318`.

Running the following command with the default settings reproduces the same adapter:

```bash
pip install ms-swift==4.5.2 bitsandbytes

python finetune/train.py --data data/train.jsonl
```

See **[finetune/README.md](finetune/README.md)** for the training data format, GPU memory requirements, and instructions for using a different base model.

## 📊 Evaluation

The evaluation pipeline is fully open source and reproducible.

<p align="center">
  <img src="assets/evaluation.webp" alt="VoiceMem Evaluation Results" width="100%">
</p>

### Run Evaluation

A benchmark can be started with a single command:

```bash
export OPENAI_API_KEY=sk-...

# Start with the small example included in the repository
# to make sure everything is set up correctly.
# 2 conversations, 5 questions.
python evaluation/run.py \
    --dataset locomo \
    --data evaluation/examples/locomo_sample.json

# Then run the full dataset.
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json
```

Example result:

```text
LoCoMo: 10 conversations · 152 questions

Score: 139/152 = 91.4%

  multi_hop     88.2%
  temporal      85.7%
  single_hop    95.1%

Median retrieval latency: 12 ms
Median retrieved memory: 298 tokens
```

Before running a full evaluation, add `--inspect` to check how the dataset is parsed.

This mode does not call the model and does not incur API costs:

```bash
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json \
    --inspect
```

During evaluation, the answering model receives **only the retrieved memories**, not the original conversation history.

If the model receives the full conversation, the benchmark becomes a reading-comprehension test rather than an evaluation of the memory system itself.

See **[evaluation/README.md](evaluation/README.md)** for the complete evaluation protocol and instructions for adding a new benchmark. Adding a benchmark only requires one file and two functions.

## Acknowledgements

We thank the following excellent open-source projects:

* [mem0](https://github.com/mem0ai/mem0) — vector memory engine
* [FunASR](https://github.com/modelscope/FunASR) — streaming ASR with `paraformer-zh-streaming`
* [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — Silero VAD, 3D-Speaker speaker verification, and fallback streaming ASR
* [intfloat/multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small) — local embeddings and slot classification

VoiceMem also uses OpenAI APIs for Chat, TTS, and Realtime functionality.

## License

VoiceMem is open source under the **Apache License 2.0**.

See [LICENSE](LICENSE) for details.

<p align="center">
  <a href="#chinese">⬆ 回到中文 / Back to top</a>
</p>
