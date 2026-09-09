#!/usr/bin/env bash
# Scarica i **modelli locali** usati da voicemem, organizzati per scopo in directory:
#
#   models/
#     vad/        silero_vad.onnx                                  Rileva "ha finito di parlare"
#     asr/        sherpa-onnx-nemotron-3.5-asr-streaming-0.6b…    ASR streaming multilingua EN/IT (Nemotron 3.5)
#     speaker/    3dspeaker_speech_eres2net_base_sv_zh-cn…onnx     Voiceprint del parlante
#     embedding/  intfloat/multilingual-e5-small                   Vettori di memoria + classificazione slot (condivisi)
#     scene/      MIT/ast-finetuned-audioset-10-10-0.4593          Scena acustica
#     emotion/    FunAudioLLM/SenseVoiceSmall                      Emozione + trascrizione fine
#     tts/        (opzionale) piper voice .onnx, richiesto solo se TTS_BACKEND=local
#
# Dopo aver scaricato questi, l'intera catena non richiede più rete (tranne per quella chiamata API del modello di risposta). Si può eseguire anche senza —
# il codice farà fallback agli ID HF, e transformers li scaricherà automaticamente al primo utilizzo.
#
# Due cose che non sono qui:
#   · Modello di risposta — adapter PEFT (180MB), agganciato a Qwen/Qwen3.6-35B-A3B, il modello base si prende a parte.
#     Aggiungi --reply-adapter per scaricarlo, vedi sotto.
#   · Qwen2.5-Omni per l'attribuzione emotiva — al momento non è disponibile una versione fine-tuned, il codice usa di default
#     Qwen/Qwen2.5-Omni-3B ufficiale, che viene scaricato automaticamente al primo utilizzo. Se hai una tua versione fine-tuned,
#     imposta export VOICEMEM_OMNI_MODEL=/tuo/percorso per puntare lì.
#
# Utilizzo (dalla root del repository):
#   bash scripts/download_models.sh                  # scarica la suite models/
#   bash scripts/download_models.sh /path/to/models  # cambia directory di destinazione
#   bash scripts/download_models.sh --reply-adapter  # scarica extra l'adapter del modello di risposta
#   VOICEMEM_FROM_UPSTREAM=1 bash scripts/download_models.sh   # cambia per scaricare da varie fonti ufficiali una alla volta
set -euo pipefail

DEST="models"
WANT_SLM=0
for arg in "$@"; do
  case "$arg" in
    --reply-adapter|--slm) WANT_SLM=1 ;;   # --slm è il vecchio nome, mantenuto per compatibilità con vecchi comandi
    *)     DEST="$arg" ;;
  esac
done

REPO="${VOICEMEM_MODELS_REPO:-zhifeixie/VoiceMem_Default_Models_Env}"
NEMOTRON_DIR="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
NEMOTRON_URL="${VOICEMEM_NEMOTRON_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NEMOTRON_DIR}.tar.bz2}"
# Nota: nel nome del repository c'è scritto Qwen25_omni, ma il contenuto è l'adapter di risposta per Qwen3.6-35B.
ADAPTER_REPO="${VOICEMEM_REPLY_ADAPTER_REPO:-${VOICEMEM_SLM_REPO:-zhifeixie/VoiceMem_SLM_Qwen25_omni}}"
mkdir -p "${DEST}"
# Rimuove eventuali ASR legacy lasciati da un download precedente.
rm -rf "${DEST}/asr/funasr-paraformer-zh-streaming" \
  "${DEST}/asr/paraformer-zh-streaming" \
  "${DEST}/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

if [ "${VOICEMEM_FROM_UPSTREAM:-0}" != "1" ]; then
  echo "[1/2] Scarico i modelli locali non-ASR da ${REPO} …"
  python3 - "${REPO}" "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    ignore_patterns=["asr/**", "**/funasr*", "**/paraformer*", "**/zipformer*"],
)   # ripresa da interruzione, esecuzione ripetuta non riscarica
PY
  mkdir -p "${DEST}/asr"
  if [ ! -f "${DEST}/asr/${NEMOTRON_DIR}/encoder.int8.onnx" ]; then
    echo "      Fonte ufficiale: Nemotron 3.5 ASR Streaming 0.6B EN/IT …"
    curl --fail --location --retry 3 "${NEMOTRON_URL}" | tar xj -C "${DEST}/asr"
  fi
else
  # Scarica uno per uno da varie fonti ufficiali pubbliche (quando HF è inaccessibile, o si vogliono verificare origini e licenze)
  REL="https://github.com/k2-fsa/sherpa-onnx/releases/download"
  mkdir -p "${DEST}"/{vad,asr,speaker,embedding,scene,emotion}

  echo "[1/3] Fonte ufficiale: VAD silero (MIT)…"
  curl -L -o "${DEST}/vad/silero_vad.onnx" "${REL}/asr-models/silero_vad.onnx"

  echo "      Fonte ufficiale: Nemotron 3.5 ASR Streaming 0.6B EN/IT (Apache-2.0, k2-fsa)…"
  if [ ! -f "${DEST}/asr/${NEMOTRON_DIR}/encoder.int8.onnx" ]; then
    curl -L "${REL}/asr-models/${NEMOTRON_DIR}.tar.bz2" | tar xj -C "${DEST}/asr"
  fi

  # Nota: il tag di release ufficiale è scritto proprio "recongition" (con errore)
  echo "      Fonte ufficiale: voiceprint 3D-Speaker ERes2Net (Apache-2.0)…"
  curl -L -o "${DEST}/speaker/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
    "${REL}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

  echo "      Fonte ufficiale: embedding / scene / emotion (repository HF originali)…"
  python3 - "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
# Scarica solo i pesi che verranno effettivamente caricati. Questi repository contengono simultaneamente safetensors / pytorch_model.bin /
# varianti quantizzate onnx / openvino, scaricare l'intero repository sono 4.2G, prendendo solo i necessari circa 1.4G — risparmi molto nel download e
# nel successivo trasferimento al repository di pubblicazione.
SKIP = ["*.bin", "onnx/*", "openvino/*", "*.tflite", "*.h5", "*.msgpack", "coreml/*"]
for kind, repo, skip in [ # embedding multilingua per memoria + slot classification
                         ("embedding", "intfloat/multilingual-e5-small", SKIP),
                         ("scene",     "MIT/ast-finetuned-audioset-10-10-0.4593", SKIP),
                         # I pesi di SenseVoice sono proprio model.pt, non possono essere esclusi con il metodo *.bin
                         ("emotion",   "FunAudioLLM/SenseVoiceSmall", ["*.onnx", "*.tflite"])]:
    print(f"        {kind} ← {repo}")
    snapshot_download(repo_id=repo, local_dir=f"{dest}/{kind}", ignore_patterns=skip)
PY
fi

if [ "${WANT_SLM}" = "1" ]; then
  echo "[2/2] Adapter modello di risposta ${ADAPTER_REPO} …"
  python3 - "${ADAPTER_REPO}" "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=f"{sys.argv[2]}/reply_adapter")
PY
  echo "      Usalo: export VOICEMEM_REPLY_ADAPTER=${DEST}/reply_adapter"
  echo "      Questo è un adapter, non un modello completo — il modello base Qwen/Qwen3.6-35B-A3B va preso separatamente, secondo la sua licenza."
  echo "      Esegui: python examples/03_simple_agent_with_voicemem_memory.py"
else
  echo "[2/2] Salta adapter modello di risposta (aggiungi --reply-adapter se serve)"
fi

echo
echo "Completato: ${DEST}/ organizzato per scopo (vad / asr / speaker / embedding / scene / emotion)."
echo "Il codice userà prima i modelli locali qui, le voci non disponibili faranno fallback automatico al download HF."
