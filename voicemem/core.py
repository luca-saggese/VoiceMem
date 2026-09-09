"""Facade di livello superiore di voicemem: una classe VoiceMem = cervello sinistro + cervello destro + percezione audio + un set di capacità intercambiabili (utils).

    Cervello sinistro  Memoria fattuale: entità + mappa cognitiva (classificazione/retrieval slot), database vettoriale mem0 sottostante
    Cervello destro  Memoria emotiva: valence-arousal per turno, attribuzione emotiva, ritratto di personalità
    Utils capacità plug-in: embedding / schema(classificazione) / entity / emotion / voiceprint / asr / memory_engine
          Ognuno ha un default integrato, passa una funzione per sostituire con la tua (modello locale, altro database vettoriale…)

    vm = VoiceMem(api_key="sk-...", mode="text_mode")
    vm.ingest("A mezzogiorno ho mangiato ramen con Alex")
    vm.search("Cosa ho mangiato a mezzogiorno?")                    # Retrieval da entrambi i cervelli
    vm.left_brain.search(...) / vm.right_brain.search(...)
    VoiceMem(embedding=lambda: MyE(), schema=lambda: MyClassifier())   # Sostituisci una capacità


La mode determina quali capacità caricare: left_brain_single / text_mode / multi_modal (con audio).

Questo file è solo l'ingresso "esposto agli altri per comprendere il sistema": VoiceMem è un thin facade, i metodi convenience minuscoli rivolti all'utente
(ingest/search/classify/preprocess/flush/test) delegano ciascuno su una riga a un'istanza interna di
Orchestrator (self._o). L'intera pipeline vera (orchestrazione Search/Ingest, metodi
utility, forwarding ai tre componenti cervello sinistro/destro/audio, SearchResult / Utils) sono nascosti in orchestrator.py.

# Se vuoi chiamare direttamente i metodi di orchestrazione maiuscoli (vm.Search / vm.Ingest / vm.Classify / vm.Flush…)
# o accedere ai metodi interni di forwarding è possibile: il __getattr__ della facade reindirizza trasparentemente a self._o.
"""

from __future__ import annotations

from pathlib import Path

from voicemem.orchestrator import Orchestrator, SearchResult, Utils
from voicemem.llm_config import MODELS

# SearchResult / Utils vengono re-esportati qui, mantenendo `from voicemem.core import ...` e
# `from voicemem import SearchResult, Utils` funzionanti.
__all__ = ["VoiceMem", "SearchResult", "Utils"]


class VoiceMem:
    """Facade di livello superiore: cervello sinistro + cervello destro + utils (capisci il sistema a colpo d'occhio). Implementazione vedi orchestrator.py.

    Lato input ``stream()`` → ``Turn`` (voicemem/stream.py), lato output ``reply()`` /
    ``reply_stream()`` (voicemem/reply.py); ``VoiceMem(reply=fn)`` per sostituire con il tuo modello.

    I metodi convenience minuscoli rivolti all'utente delegano ciascuno su una riga a ``self._o`` interno (un'istanza di ``Orchestrator
    ``); ``left_brain`` / ``right_brain`` / ``utils`` puntano direttamente ai componenti reali. I parametri del costruttore vengono
    passati a ``Orchestrator``: ``mode`` usa mode, sovrascritture di capacità (``embedding`` / ``schema`` /
    ``memory_engine`` ecc.) e ``enable_*`` / ``embedder`` / ``vector_store`` / ``classifier``
    vengono tutti passati tramite ``**kw``, semanticamente identici a ``Orchestrator.__init__``.
    """

    #: Alias esterni di mode → nomi interni. Il README usa terminologia user-facing ("normal" = tutto incluso,
    #: "leftbrain_only" = solo memoria fattuale), i nomi interni descrivono quali util vengono caricate.
    MODE_ALIASES = {
        "normal":         "multi_modal",
        "leftbrain_only": "left_brain_single",
        "text":           "text_mode",
    }

    def __init__(self, api_key=None, mode="text_mode", memory_root=None,
                 user_id="voice_user", base_url=None, reply=None,

                 openai_key=None, top_k=5, space=None, models=None,
                 memory_language=None, model_name=None, **kw):
        # Lingua del testo salvato nella memoria: "en" (default) o "zh". Vedi voicemem/lang.py.
        # I nomi degli slot e le 8 emozioni canoniche sono enumerazioni interne, non influenzate.
        # La lingua si applica allo **spazio corrispondente a questa istanza**, non è globale al processo — vedi resolve_for_space.
        # openai_key è il vecchio nome di api_key, equivalente, mantenuto per compatibilità. Il nuovo codice usa api_key.
        # Ogni modello può essere selezionato separatamente: {"chat": ..., "reply": ..., "embedding": ...,
        # "tts": ..., "realtime": ...}. I ruoli omessi seguono env / valori default come prima.
        # Nota: questa è un'impostazione **a livello di processo**, non privata di questa istanza: i componenti sinistro/destro fanno lazy loading,
        # leggono dalla tabella globale solo quando servono. Se apri un secondo VoiceMem nello stesso processo con modelli diversi, il primo
        # cambia anche lui (MODELS.update stampa una riga quando succede). Per isolamento rigoroso usa processi separati.
        # I valori vengono letti dalla tabella globale solo al momento dell'uso. Se apri un secondo VoiceMem nello stesso processo con modelli diversi, il primo
        # cambia anche lui (MODELS.update stampa una riga quando succede). Per isolamento rigoroso usa processi separati.
        if model_name:
            models = {**(models or {}), "chat": model_name}
        if base_url:
            import os
            os.environ["OPENAI_BASE_URL"] = base_url
        if models:
            MODELS.update(models)
        # space: una directory per set di memorie, situato in ./voicemem_memoryspace/<space>/.
        # Se non passato usa "demo". Se memory_root è esplicitamente passato lo usa così com'è (la valutazione richiede un database indipendente per ogni conversazione).

        self._o = Orchestrator(api_key=api_key or openai_key,
                               mode=self.MODE_ALIASES.get(mode, mode),
                               memory_root=memory_root, space=space,
                               user_id=user_id, base_url=base_url, **kw)

        # Il layer di risposta è una questione a livello facade (il layer di orchestrazione arriva solo ai risultati della memoria), quindi reply non viene passato oltre.
        # None → al primo utilizzo fa fallback al provider openai integrato, vedi _reply_fn.
        #
        # Perché reply non è il decimo slot di capacità (tts è uno slot di capacità, entrambi sono sul lato output): il valore di uno slot di capacità
        # può essere sia una factory che un oggetto già costruito, si distingue da "se è una funzione/classe" (Utils.get).
        # Ma **il provider di risposta è esso stesso una funzione** — `VoiceMem(reply=my_async_gen_fn)`
        # ciò che viene passato verrebbe chiamato come factory una volta, causando errori. Questa ambiguità è unica del layer di risposta,
        # quindi segue separatamente la strada di normalize(), non entra nella tabella delle capacità.
        self._reply_src = reply
        self._reply_norm = None
        self._top_k = top_k                  # Quante voci prendere di default da search()
        # La directory dello spazio viene determinata solo ora (Orchestrator risolve space/memory_root), quindi messa dopo.
        from voicemem.lang import resolve_for_space
        resolve_for_space(self._o._memory_root, memory_language)

        self.mode = self._o.mode
        self.utils = self._o.utils
        self.left_brain = self._o._left      # Componente reale
        self.right_brain = self._o._right

    @classmethod
    def from_config(cls, config: dict) -> "VoiceMem":
        """Costruzione dichiarativa: un unico dict config per configurare tutti i modelli locali/api (modello mem0).


        Ogni componente è scritto come ``{"provider": ..., "config": {...}}``, aprendo il dict sai se ogni modello
        usa locale o api. Questo è uno strato di zucchero sopra il meccanismo di iniezione esistente ``VoiceMem(embedding=fn, schema=fn, …)
        ``, i metodi di costruzione esistenti funzionano normalmente. La tabella di mappatura dei provider vedi
        ``voicemem.config``.::


            vm = VoiceMem.from_config({
                "mode": "multi_modal",
                "embedding": {"provider": "local"},   # Vettori memoria usano E5 locale
                "slots":     {"provider": "local"},   # Classificazione slot usa E5 locale (0 LLM)
                "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            })
        """
        from voicemem.config import build_kwargs
        return cls(**build_kwargs(config))

    # ── Metodi convenience user-facing (ciascuno delega su una riga a Orchestrator) ──────────────────────────

    def ingest(self, text=None, audio=None, **kw):
        """Salva una frase. ``ingest("testo")`` salva testo; ``ingest(audio="x.wav")`` quando passi solo audio
        prima trascrivi localmente poi salva (lo stesso audio passa comunque attraverso voiceprint/scena/percezione emotiva). Se passi entrambi usa il testo fornito."""
        if audio is not None:
            audio = sample_audio(audio)   # Import locale: __init__ importa core, importare a livello globale causerebbe ciclo
            audio = sample_audio(audio)
        if text is None:
            if audio is None:
                raise ValueError("ingest() deve ricevere o text o audio")
            text = self.transcribe(audio)
        # Senza audio non c'è voiceprint, il parlante può essere solo il proprietario dell'account — usa l'id convenzionale "user"
        # (vedi voice_input_to_messages), altrimenti si finisce con l'etichetta difensiva preparata per "voiceprint non verificato",
        # e ogni fatto salvato diventa "Speaker sconosciuto Speaker 0 è vegetariano".
        # Con audio mantieni il default del layer di orchestrazione, lascia che il riconoscimento voiceprint decida il parlante (in scenari multi-persona non si può assumere sia il proprietario).
        if audio is None:
            kw.setdefault("speaker", "user")
        return self._o.Ingest(text, audio_path=audio, **kw)

    def transcribe(self, audio) -> str:
        """Trascrivi un file audio completo in testo (riutilizza l'ASR streaming da utils, non scarica modelli aggiuntivi)."""
        from voicemem.utils.audio.stream_io import transcribe_file
        return transcribe_file(self.utils.get("asr"), audio)

    def search(self, query, **kw):
        kw.setdefault("top_k", self._top_k)
        return self._o.Search(query, **kw)

    def classify(self, query):                return self._o.Classify(query)
    def preprocess(self, text, audio=None):   return self._o.preprocess(text, audio_path=audio)
    def flush(self):                          return self._o.Flush()

    def warmup(self, *, audio: bool = True, verbose: bool = True) -> None:
        """Carica tutti i modelli locali prima, non far aspettare alla prima chiamata.

        I modelli sono caricati lazy: senza warmup la prima ``ingest(audio=...)`` richiede altri venti secondi
        (E5 locale ~1.7s, FunASR ~6.5s, suite percezione ~16s), e questi secondi cadono esattamente quando
        l'utente parla per la prima volta — il punto dove non si dovrebbe mai bloccare. Il web demo lo fa sempre,
        ora è spostato qui, così tutti i demo e gli script scritti manualmente possono chiamarlo con una riga.

        ``audio=False`` carica solo il percorso testo (non serve ASR/percezione per leftbrain_only puro).
        ``verbose=False`` non stampa nulla.
        Le chiamate ripetute sono sicure: quando i modelli sono già caricati ogni passo è solo un empty run economico.
        """
        import sys
        import time

        total = 4 if audio else 1
        # Solo i terminali veri mostrano la barra di progresso: quando reindirizzi a file/pipeline \r si confonde in una riga
        bar_ok = verbose and sys.stdout.isatty()
        done = [0]

        def draw(label, finished=False):
            if not verbose:
                return
            if not bar_ok:
                if finished:
                    print(f"[warmup] {label}", flush=True)
                return
            width = 24
            filled = int(width * done[0] / total)
            bar = "█" * filled + "░" * (width - filled)
            pct = int(100 * done[0] / total)
            end = "\n" if done[0] >= total else ""
            # 31 = rosso
            sys.stdout.write(f"\r\033[31m{bar}\033[0m {pct:3d}%  {label:<28}{end}")
            sys.stdout.flush()

        def step(name, fn):
            draw(f"Caricamento {name} …")
            t0 = time.time()
            try:
                fn()
                label = f"{name} {time.time() - t0:.1f}s"
            except Exception as e:                 # Dipendenze opzionali mancanti / modello non scaricato
                label = f"{name} saltato ({type(e).__name__})"
            done[0] += 1
            draw(label if done[0] < total else "Modelli pronti", finished=True)

        step("embedding / classificazione slot", lambda: self.classify("Ciao"))
        if not audio:
            return

        import numpy as np

        def warm_asr():
            asr = self.utils.get("asr")
            asr.feed(np.zeros(9600, dtype=np.float32))     # Avvia il modello ed esegui realmente un blocco
            asr.reset()

        step("ASR", warm_asr)
        step("VAD", lambda: self.utils.get("vad").is_speech(np.zeros(512, dtype=np.float32)))

        # La suite percezione (scene AST / voiceprint 3D-Speaker / emozione SenseVoice) richiede un file reale.
        # Nei test la prima preprocess richiede 2120ms tutti nel caricamento modelli, poi si stabilizza a 340-410ms.
        def warm_perceive():
            import tempfile
            import soundfile as sf
            from pathlib import Path
            p = Path(tempfile.gettempdir()) / "voicemem_warmup.wav"
            sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
            try:
                self.preprocess("Warmup", audio=str(p))
            finally:
                p.unlink(missing_ok=True)

        step("Percezione (scena / voiceprint / emozione)", warm_perceive)

    def stream(self, **kw):
        """Percorso input streaming: fornisci blocchi audio / testo → quando hai finito ottieni Turn (risultati memoria). Vedi voicemem/stream.py."""
        from voicemem.stream import VoiceStream
        return VoiceStream(self, **kw)

    # ── Layer risposta (lato output): due percorsi, un'unica interfaccia, vedi voicemem/reply.py ────────────────────

    def _reply_fn(self):
        """Normalizzazione lazy: qualsiasi forma passata da VoiceMem(reply=fn) → funzione async generator unificata.
        Se non passato usa il provider openai integrato (il client viene creato solo alla prima chiamata, puoi importare senza passare key)."""
        if self._reply_norm is None:
            from voicemem.reply import normalize, openai_reply
            self._reply_norm = normalize(self._reply_src or openai_reply())
        return self._reply_norm

    def reply_stream(self, turn_or_text, memory_context=""):
        """Risposta streaming: ``async for delta in vm.reply_stream(turn)``.

        Il primo parametro può essere direttamente ``Turn``/``StreamState`` (decompone automaticamente text e memory_context),
        oppure puoi passare del testo + memory_context renderizzato manualmente.

        Questa frase detta viene registrata automaticamente al layer memoria (``capture`` → ``remember_reply``), alla prossima
        ``ingest()`` include già questa metà dell'agent, il chiamante non deve cambiare nessuna riga.
        """
        from voicemem.reply import capture, unpack
        text, ctx = unpack(turn_or_text, memory_context)
        return capture(self._reply_fn()(text, ctx),
                       lambda answer: self._o.remember_reply(text, answer))

    async def reply(self, turn_or_text, memory_context=""):
        """Risposta completa: ``answer = await vm.reply(turn)``. Internamente concatena reply_stream."""
        return "".join([d async for d in self.reply_stream(turn_or_text, memory_context)])

    def test(self):
        """Avvia auto-test: misura solo gli util richiesti da questa mode, stampa una tabella delle velocità a 4 livelli."""
        from voicemem.startup_check import run_util_report
        return run_util_report(self.utils)

    def __getattr__(self, name):
        # I vecchi metodi di orchestrazione maiuscoli (Search/Ingest/Classify/Flush…) e i metodi interni di forwarding
        # vengono trasparentemente delegati all'implementazione dell'orchestratore; __getattr__ viene attivato solo quando la ricerca regolare degli attributi fallisce, quindi self._o /
        # utils / left_brain / right_brain — questi attributi già impostati non arrivano qui.
        return getattr(self.__dict__["_o"], name)
