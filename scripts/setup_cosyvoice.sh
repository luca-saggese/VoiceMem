#!/usr/bin/env bash
# Installazione riproducibile del source tree CosyVoice + Matcha-TTS.
# Il modello e i pesi restano esclusi da Git e vengono scaricati separatamente.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/CosyVoice"
REF="${COSYVOICE_COMMIT:-main}"

if [ ! -d "${DEST}/.git" ]; then
  mkdir -p "$(dirname "${DEST}")"
  git clone https://github.com/FunAudioLLM/CosyVoice.git "${DEST}"
fi

git -C "${DEST}" fetch --tags origin
if git -C "${DEST}" rev-parse --verify "${REF}^{commit}" >/dev/null 2>&1; then
  git -C "${DEST}" checkout --detach "${REF}"
else
  echo "Commit/tag CosyVoice non trovato: ${REF}" >&2
  exit 1
fi

if [ -f "${DEST}/requirements.txt" ]; then
  python3 -m pip install -r "${DEST}/requirements.txt"
fi
if [ -f "${DEST}/third_party/Matcha-TTS/requirements.txt" ]; then
  python3 -m pip install -r "${DEST}/third_party/Matcha-TTS/requirements.txt"
fi

printf '%s\n' "${REF}" > "${ROOT}/third_party/COSYVOICE_VERSION"
echo "CosyVoice installato in ${DEST}"
echo "Versione: ${REF}"
echo "Imposta VOICEMEM_COSYVOICE_MODEL e VOICEMEM_COSYVOICE_REF_AUDIO prima di avviare il web."
