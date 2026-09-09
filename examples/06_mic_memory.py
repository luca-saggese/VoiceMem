"""Solo ascolto senza risposta: microfono → trascrizione → retrieval memoria. Nessuna risposta LLM, nessun TTS.

    export OPENAI_API_KEY=sk-...
    python examples/06_mic_memory.py

Dici una frase, vedrai tre cose accadere in sequenza:

    [Ascolto]  Trascrizione in tempo reale, le parole appaiono mentre parli ancora
    [Ricerca]  Nel momento in cui finisci, la memoria rilevante **è già pronta** — il retrieval è stato
               completato in background durante il tuo parlato (prefetch speculativo 0–500ms), non occupa tempo dopo che hai finito
    [Salvataggio]  Questo turno viene scritto nel database memoria, la prossima volta potrà essere recuperato

Per vedere se ha davvero memorizzato: dì "Sono allergico alle arachidi", poi dopo qualche altra frase chiedi "Cosa non posso mangiare?".

Ctrl-C per uscire.
"""
import asyncio
import os
import queue
import sys
from pathlib import Path

import sounddevice as sd

from voicemem import VoiceMem

SR = 16000          # Campionamento del microfono
BLOCK = 512         # Numero di campioni per blocco: 32ms @16k, allineato alla lunghezza del frame VAD

vm = VoiceMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},   # Vettori memoria: locale, 0 rete
    "slots": {"provider": "local"},       # Classificazione slot: locale, 0 LLM
    "api_key": os.environ["OPENAI_API_KEY"],   # Usato solo nel lato scrittura per estrarre fatti
    # Libreria separata: E5 locale ha 384 dimensioni, mescolandolo con la libreria default (OpenAI 1536 dimensioni) si otterrà un errore
    # shapes (n,384) and (1536,) not aligned
    # shapes (n,384) and (1536,) not aligned
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
})


def show_partial(text):
    print(f"\r[Ascolto] {text}", end="", flush=True)


async def main():
    vm.warmup()

    # Il callback di sounddevice gira nel suo thread, non può fare await direttamente. Usa una coda per passare i dati,
    # lascia che il loop eventi li prenda da qui — nel callback fai solo搬运 (trasferimento), non bloccare assolutamente, altrimenti perdi audio.
    blocks: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        blocks.put(bytes(indata))

    stream = vm.stream(src_rate=SR, on_partial=show_partial)

    with sd.RawInputStream(samplerate=SR, blocksize=BLOCK, dtype="int16",
                           channels=1, callback=on_audio):
        print("Parla (Ctrl-C per uscire)\n", flush=True)
        while True:
            pcm = await asyncio.to_thread(blocks.get)
            st = await stream.feed(pcm)
            if st.state != "turn_over":
                continue

            print(f"\r[Ascolto] {st.transcript}")

            left = st.result_leftbrain or []
            right = st.result_rightbrain or []
            if left or right:
                print("[Ricerca] Memoria già pronta al momento in cui hai finito:")
                for m in left:
                    print(f"       Cervello sinistro  {m}")
                for m in right:
                    print(f"       Cervello destro  {m}")
            else:
                print("[Ricerca] Nessuna memoria rilevante ancora (il database è vuoto, parla un po' di più)")

            # 写入是秒级的，丢线程别挡住麦克风
            asyncio.create_task(asyncio.to_thread(vm.ingest, st.transcript))
            print("[Salvataggio] Scritto, la prossima volta sarà recuperato\n", flush=True)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n再见。")
    sys.exit(0)
