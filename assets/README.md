# assets

Materiale audio per demo e illustrazioni.

## input.wav

Utilizzato nel primo esempio del README (`vm.ingest(audio="input.wav")`).
Contiene il testo "Sono vegetariano, sono allergico alle noci.", durata 5.8 secondi,
PCM16 mono 16 kHz, sintetizzato con OpenAI TTS. Sono stati lasciati 1 secondo di
silenzio alla fine — il percorso streaming richiede 0.5 secondi di silenzio continuo
per capire che l'utente ha finito di parlare, altrimenti `turn_over` non arriverebbe mai se l'audio si interrompe esattamente quando finisce di parlare.

## question.wav

L'audio usato nella sezione streaming del README, passato a `vm.stream()`. Contiene
il testo "Quali sono le mie restrizioni alimentari?", durata 3.2 secondi, PCM16 mono
16 kHz, sintetizzato con macOS `say -v Tingting`.

**È una domanda** — è proprio questo che viene dimostrato nello streaming: la memoria
viene cercata prima ancora che l'utente abbia finito di parlare, e l'ultima chiamata
`ingest()` non memorizzerà alcun fatto nuovo (poiché la frase non contiene informazioni
nuove sull'utente), risultando in `facts_count` 0. Anche qui sono stati lasciati oltre
1 secondo di silenzio alla fine, affinché `turn_over` possa essere rilevato correttamente.

## speech.wav

Input predefinito per `examples/02_streaming.py`.
Contiene il testo "Mi piacciono i macaron", durata 7.7 secondi, PCM16 mono 16 kHz.

## cafe_song.wav

"Quella canzone sentita al caffè" — quando viene chiesta nel web demo, questo audio originale viene riprodotto.

Durata 15 secondi, PCM16 mono 16 kHz. Composto dalla miscela di due fonti:

| Strato | Fonte | Licenza |
|---|---|---|
| Musica | *Chili Pepper* — Fred Longshaw, registrazione jazz di pianoforte del 1927 ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Chili_Pepper_by_Fred_Longshaw_(1927,_Jazz_piano).opus)) | Pubblico dominio (registrazione del 1927) |
| Ambiente | *Restaurant ambience* ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Restaurant_ambience.ogg)) | Vedi pagina file su Commons |

Parametri di missaggio (la musica viene abbassata a strato di sottofondo, come se provenisse dagli altoparlanti del locale):

```bash
ffmpeg -ss 8 -t 15 -i music.opus -stream_loop -1 -t 15 -i amb.ogg \
  -filter_complex "[0:a]volume=0.55,highpass=f=120,lowpass=f=6500[m];\
[1:a]volume=1.0[a];[m][a]amix=inputs=2:duration=first:normalize=0,\
dynaudnorm=p=0.7,alimiter=limit=0.95[out]" \
  -map "[out]" -ac 1 -ar 16000 -c:a pcm_s16le cafe_song.wav
```

Per usare la tua registrazione: basta sovrascrivere questo file (l'archivio registra il percorso), ma **si consiglia di rieseguire
ingest** — i tag `tune:` / `scene:` vengono calcolati dall'audio, cambiando contenuto i tag non corrisponderebbero più.
