"""voicemem.utils — Layer di capacità/strumenti che supporta i due cervelli.

- ``audio/``    Percezione audio-native: ASR, voiceprint del parlante, VAD emotiva, ambiente/acustica.
- ``common/``   Strumenti cross-cutting: tracciamento sessioni, turn_id, adattamento accesso vocale, utility comuni grafo, log costi, configurazione.
- ``fusion/``   Fusione output sinistro/destro + orchestrazione risposta (un'orchestrazione di livello superiore indipendente dal motore principale).

Questi sono strumenti di "come percepire, come coordinare", non la memoria stessa — la memoria sinistro/destro si trova in
``voicemem.leftbrain`` / ``voicemem.rightbrain``. I sotto-moduli vengono importati lazy su richiesta (vedi mappa PEP 562 nel file
``voicemem/__init__.py``), ``import voicemem`` non avvierà torch/sherpa.
"""
