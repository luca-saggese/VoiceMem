"""Testo → voce. Il layer di risposta produce solo testo (vedi ``voicemem/reply.py``), qui avviene la conversione vocale, è un layer opzionale.

Quattro backend integrati, tutti producono **24kHz mono PCM16**: default API OpenAI; ``local`` / ``piper``
usano piper offline; ``voxcpm`` usa VoxCPM2; ``breeze`` si connette al servizio streaming Breeze TTS 2
(tono di voce con linguaggio naturale, vedi ``BreezeTTS``; licenza non commerciale, quindi non è il default).

**Questo è il nono slot intercambiabile** (gli altri otto vedi ``voicemem/utils/defaults.py``). Il contratto ha un solo metodo::

    class MyTTS:
        async def stream(self, text: str):     # produce bytes 24kHz mono PCM16 in modo asincrono
            ...

    vm = VoiceMem(tts=lambda: MyTTS())                      # Scrittura A: iniezione
    vm = VoiceMem.from_config({"tts": {"provider": "voxcpm"}})   # Scrittura B: dichiarativa

Il percorso core non lo tocca — il sistema memoria arriva solo al testo, l'audio è compito del chiamante. Quindi ``tts`` non è in
``_NEED`` (non viene avviato da warmup), chi vuole audio fa ``vm.utils.get("tts")``;
 gli utenti senza piper / voxcpm installati non sono influenzati.

I backend che supportano informazioni di allineamento possono produrre ``TimedAudioChunk``; gli timestamp usano campioni relativi all'inizio del segmento.
I backend puri ``bytes`` esistenti non richiedono modifiche.


``speak_stream()`` è "sintetizza mentre genera": quando viene completata una frase viene inviata alla sintesi, non si aspetta che tutto il testo sia generato —
se si sintetizza dopo aver generato tutto, le parole sono già finite ma l'audio non è ancora partito (nei test la prima frame del TTS richiede ~1.2s).
"""
from __future__ import annotations

import asyncio
import os

import numpy as np

from voicemem.utils.audio.stream_io import resample
from voicemem.audio_timing import TimedAudioChunk, TextTimestamp
from voicemem.llm_config import resolve_model


#: Compatibile con vecchi nomi. La vera analisi avviene in OpenAITTS.__init__ (vedi llm_config) —
#: prima si leggeva env all'import, se impostavi env dopo l'import rimaneva silenziosamente inefficace.
TTS_MODEL = resolve_model(role="tts")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "cosyvoice3")   # cosyvoice3 | openai | local | voxcpm
#: Timbro vocale. Prima questo valore era scritto hardcoded nella funzione di sintesi, gli utenti di llm_tts volevano cambiarlo ma dovevano modificare il codice sorgente —
#: anche le implementazioni default integrate dovrebbero essere configurabili, altrimenti "intercambiabile" significa solo sostituire l'intero backend.
TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "alloy")
#: Come leggere (velocità, enfasi, pause). gpt-4o-mini-tts supporta instructions, è dove controllare il tono.
#: Se vuoto non passa questo parametro — i modelli vecchi (tts-1) non lo riconoscono, passandolo causerebbe un errore.
TTS_INSTRUCTIONS = os.environ.get("OPENAI_TTS_INSTRUCTIONS", "")
#: Ha senso solo durante "generazione timbri candidati" (stesso seed + stesso testo = stessa persona, utile per scegliere).
#: **Non garantisce coerenza del timbro tra frasi**, a bloccare il timbro è ref_audio, vedi documentazione BreezeTTS.
BREEZE_SEED = int(os.environ.get("VOICEMEM_BREEZE_SEED", "42"))
SAMPLE_RATE = 24000

# Come tagliare i segmenti per il TTS determina quanto tempo ci vuole per la prima voce. Nei test la latenza della prima frame di gpt-4o-mini-tts cresce con la lunghezza del testo:
# 8 caratteri 615ms / 25 caratteri 902ms / 100 caratteri 1318ms — quindi **il primo segmento deve essere il più corto possibile** (prima voce in uscita),
# i segmenti successivi possono essere più lunghi (meno chiamate, tono coerente).
_SENT_END  = "。！？!?…\n"          # Fine frase: punto normale di taglio
_SOFT_END  = "，,、；;：: "          # Pausa interna: usato solo per il primo segmento, per far uscire la prima voce prima
#
# I numeri qui sotto erano originariamente hardcoded, calibrati sulla latenza della prima frame di gpt-4o-mini-tts. Cambiando backend devi ricalibrare:
# ogni segmento è **sintetizzato indipendentemente**, il contorno tonale non si estende tra segmenti, quindi più segmenti ci sono, più evidenti sono le giunture —
# all'ascolto sembra "frase per frase incollata" invece che parlato continuo. Tagliare finemente è scambiare fluidità per latenza della prima frame,
# quale testa vale di più dipende da quanto è veloce il backend.
_FIRST_MIN = int(os.environ.get("VOICEMEM_TTS_FIRST_MIN", "6"))
_FIRST_MAX = int(os.environ.get("VOICEMEM_TTS_FIRST_MAX", "20"))
_SENT_MIN  = int(os.environ.get("VOICEMEM_TTS_SENT_MIN", "12"))
_SENT_MAX  = int(os.environ.get("VOICEMEM_TTS_SENT_MAX", "60"))
#: Se il primo segmento può essere spezzato alla virgola. Default sì — serve per far uscire la prima voce prima. Ma questo **divide una frase in due sintesi indipendenti**,
#: la giunzione cade esattamente nel mezzo della frase, che è il tipo più sgradevole da ascoltare. Quando il backend ha latenza della prima frame sufficientemente bassa, imposta 0,
#: così tutti i segmenti cadono alla fine delle frasi, senza divisioni interne.
_FIRST_SOFT = os.environ.get("VOICEMEM_TTS_FIRST_SOFT", "1") != "0"



def cut_point(buf: str, first: bool) -> bool:
    """Questo segmento è sufficiente per inviare alla sintesi."""
    s = buf.strip()
    if not s:
        return False


    if first:                                  # Prima voce: anche la virgola conta, se proprio non ci sono altre opzioni taglia per lunghezza
        ends = _SENT_END + _SOFT_END if _FIRST_SOFT else _SENT_END
        return (len(s) >= _FIRST_MIN and s[-1] in ends) or len(s) >= _FIRST_MAX
    return (len(s) >= _SENT_MIN and s[-1] in _SENT_END) or len(s) >= _SENT_MAX


# ── Backend integrati ──────────────────────────────────────────────────────────

class BaseTTS:
    """Guscio comune dei backend integrati: le sottoclassi scrivono solo ``_raw()``, l'allineamento dei campioni viene fatto qui uniformemente.

    **Taglia ai confini dei campioni**: lo stream http è tagliato per pacchetti di rete, nei test su 69 blocchi 62 hanno byte dispari,
    mentre un campione PCM16 occupa 2 byte — il consumatore ``Int16Array``/``np.frombuffer" va in errore con lunghezza dispari
    e quel blocco audio intero viene perso. Qui si lascia mezzo campione che attraversa i blocchi al blocco successivo, garantendo che ogni blocco prodotto
    contenga campioni completi. Puoi scrivere il tuo backend senza ereditarlo, basta che ``stream()`` produca interi campioni.
    """

    async def stream(self, text: str, instruction: str | None = None):
        """``instruction``: **come leggere questa round**. Lo strato di percezione giudica l'emozione ogni turno e deve entrare nel suono,
        mentre l'emozione cambia turno per turno, scriverla sull'istanza la renderebbe una costante globale. Passa None per usare il default dell'istanza.
        I backend che non lo supportano (piper / voxcpm) possono ignorarlo."""
        tail = b""
        async for chunk in self._raw(text, instruction):
            timed = chunk if isinstance(chunk, TimedAudioChunk) else None
            raw = timed.pcm if timed is not None else chunk
            buf = tail + raw
            cut = len(buf) & ~1                     # Arrotonda per difetto al pari
            tail = buf[cut:]
            if cut:
                if timed is None:
                    yield buf[:cut]
                else:
                    yield TimedAudioChunk(
                        pcm=buf[:cut], timestamps=timed.timestamps,
                        sample_rate=timed.sample_rate)
        if tail:
            yield tail + b"\x00"                    # Completa l'ultimo mezzo campione

    def _raw(self, text: str, instruction: str | None = None):
        raise NotImplementedError


class OpenAITTS(BaseTTS):
    """API online: OpenAI TTS (default gpt-4o-mini-tts), response_format=pcm è 24k PCM16.

    ``base_url`` di default **non segue** ``VoiceMem(base_url=...)``: quello di solito punta a un servizio
    LLM/embedding self-hosted, che probabilmente non ha ``/audio/speech``, passandolo causerebbe un errore solo quando si tenta l'audio.
    Per cambiare endpoint specificalo qui esplicitamente (o ``OPENAI_TTS_BASE_URL``).
    """

    def __init__(self, model=None, voice=None, instructions=None,
                 api_key=None, base_url=None):
        self.model = resolve_model(model, "tts")
        self.voice = voice or TTS_VOICE
        self.instructions = TTS_INSTRUCTIONS if instructions is None else instructions
        self._key = api_key
        self._base = base_url or os.environ.get("OPENAI_TTS_BASE_URL") or None
        self._client = None

    def _cli(self):
        """Il client viene creato solo al primo utilizzo, ``import voicemem.tts`` non richiede quindi una key."""
        if self._client is None:
            from openai import AsyncOpenAI
            kw = {}
            if self._key:
                kw["api_key"] = self._key
            if self._base:
                kw["base_url"] = self._base
            self._client = AsyncOpenAI(**kw)
        return self._client

    async def _raw(self, text, instruction=None):
        kw = {"model": self.model, "voice": self.voice,
              "input": text, "response_format": "pcm"}
        ins = instruction or self.instructions
        if ins:
            kw["instructions"] = ins
        async with self._cli().audio.speech.with_streaming_response.create(**kw) as resp:
            async for chunk in resp.iter_bytes():
                yield chunk


class PiperTTS(BaseTTS):
    """Modello piccolo offline: piper (onnx puro offline, cinese e inglese). Installa: pip install piper-tts.

    ``model`` punta al .onnx della voce (default legge ``VOICEMEM_TTS_MODEL``). Per cambiare con kokoro /
    edge-tts ecc., scrivi una classe seguendo questo modello — fuori riconosce solo ``stream()``.
    """

    def __init__(self, model=None):
        self.model = resolve_model(model, "tts", default=None)
        self._voice = None

    def _load(self):
        if self._voice is None:
            if not self.model:
                raise ValueError(
                    "Il backend piper richiede un file voice: imposta VOICEMEM_TTS_MODEL per puntare al .onnx,\n"
                    'oppure nel config dai {"provider": "piper", "config": {"model": "…/x.onnx"}}')
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.model)   # L'api di piper cambia con le versioni, vedi la sua documentazione
        return self._voice

    async def _raw(self, text, instruction=None):
        v = self._load()                      # piper non ha un ingresso per il tono, instruction viene ignorato
        sr = getattr(getattr(v, "config", None), "sample_rate", 22050)
        for raw in v.synthesize_stream_raw(text):       # Generatore sincrono, bytes int16 @ sr
            f = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
            out = resample(f, src=sr, dst=SAMPLE_RATE)  # Uniforma a 24k
            yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class VoxCPMTTS(BaseTTS):
    """离线大模型：VoxCPM2（2B，中英+多语）。装：pip install voxcpm。
    ``model`` 可指向本地目录，缺省用 HF 上的 openbmb/VoxCPM2（走本地缓存）。"""

    def __init__(self, model=None):
        self.model = resolve_model(model, "tts", default=None) or "openbmb/VoxCPM2"
        self._m = None

    def _load(self):
        if self._m is None:
            from voxcpm import VoxCPM
            self._m = VoxCPM.from_pretrained(self.model, load_denoiser=False)
        return self._m

    async def _raw(self, text, instruction=None):
        m = self._load()                      # voxcpm 同上
        sr = m.tts_model.sample_rate
        for f in m.generate_streaming(text=text):
            out = resample(np.asarray(f, np.float32).reshape(-1), src=sr, dst=SAMPLE_RATE)
            yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class BreezeTTS(BaseTTS):
    """Breeze TTS 2（breezeblue-ai/breeze-tts）的流式服务客户端。

    它**是个服务、不是能 import 的库**：GPU 机器上起

        python -m breeze_infer.api <model_path> --host 0.0.0.0 --port 7860

    这边只是个 http 客户端，所以 VoiceMem 这侧不引入任何重依赖。要 Linux +
    NVIDIA（约 7.7 GiB 显存，建议 12GB），macOS 跑不了——``base_url`` 一般指向
    另一台机器（跟 examples/04_all_local_l40s.py 是同一个架构）。

    **不是默认后端，也不该设成默认**：代码是 Apache 2.0，但权重走 BreezeBlue 的
    研究/非商用许可，商用要 RESONIA, INC. 书面授权。设成默认等于把这个限制推给
    每一个 VoiceMem 用户。

    路径跟 OpenAI 一样是 ``/v1/audio/speech``，但收的是 form-data
    （``text`` / ``instruction`` / ``cfg_scale`` / ``ref_audio`` / ``ref_text`` /
    ``seed``），不是 JSON 的 ``input`` / ``voice``——所以不能拿 OpenAITTS 指过去。

    ``instruction`` 是用自然语言指挥语气的地方（"慢一点，说到难过的事压低声音"），
    这也是选它的理由：OpenAI realtime 那边只能在人设里夹一段文字，模型经常不理。
    正文里还能写发声事件：英文 ``(sigh)``、中文 ``[笑]``。

    **三种模式，别用错**（服务端按给没给 ref_audio 走两套不同模板）：

    ===============  ==================================  ==================
    模式             参数                                 音色
    ===============  ==================================  ==================
    Voice Design     instruction                          **每次都变**
    Voice Clone      ref_audio + ref_text                 固定
    Voice Direction  ref_audio + ref_text + instruction   固定 + 可指挥语气
    ===============  ==================================  ==================

    对话场景**必须给 ref_audio**（Voice Direction）。只给 instruction 的话说话人是
    跟文本一起生成出来的，逐句合成时每句话都会换一个人——这是实测踩到的。

    参考音频不用现录：先用 Voice Design 随机生成几段，挑一个顺耳的存下来当永久
    参考即可。要求 5~10 秒、单人、干净；``ref_text`` 必须是它的逐字转写，错了音色会飘。

    另外服务端 ``--fast-all`` 跟 ref_audio 这条路**不兼容**：text encoder 的 CUDA 图
    只按 warmup 时那几个形状捕获过，带参考音频的输入形状不在里面，会直接抛
    "text encoder CUDA graph (4, 32) was not declared in the warmup profile"。
    起服务时别加这个参数。
    """

    def __init__(self, base_url=None, instruction=None, cfg_scale=None,
                 ref_audio=None, ref_text=None, seed=None, timeout=60.0, model=None):
        self.base_url = (base_url or os.environ.get("VOICEMEM_BREEZE_URL")
                         or "http://127.0.0.1:7860").rstrip("/")
        self.instruction = instruction or os.environ.get("VOICEMEM_BREEZE_INSTRUCTION") or ""
        if cfg_scale is None:
            env_cfg = os.environ.get("VOICEMEM_BREEZE_CFG_SCALE")
            # 官方示例里 instruction 都配 cfg_scale=4（指令强度）；不给的话基本不跟指令。
            cfg_scale = float(env_cfg) if env_cfg else (4 if self.instruction else None)
        self.cfg_scale = cfg_scale
        # 参考音频这两项也给环境变量入口：web demo 的 --config 是**整体替换**内置
        # CONFIG 的，为了配一个 ref_audio 去写整份 json，很容易漏掉里面的人设
        # （reply.llm.config.system）——漏了就既没人设也不说中文。用环境变量配就
        # 不用碰那份 CONFIG。
        self.ref_audio = ref_audio or os.environ.get("VOICEMEM_BREEZE_REF_AUDIO") or None
        self.ref_text = ref_text or os.environ.get("VOICEMEM_BREEZE_REF_TEXT") or None
        # seed 只决定 voice design 从哪个随机点起步，**它锁不住音色**：说话人是跟
        # 文本一起自回归生成出来的，文本变了采样轨迹就变，同一个 seed 照样长出
        # 另一个嗓子。而这边是逐句合成的（speak_stream 攒够一句发一次请求），一段
        # 回复要发好几次——实测就是同一段话里每句话换一个人说。
        # 想固定音色只有 ref_audio 一条路，见类文档里的三种模式。
        # seed 留着是为了让「生成候选音色」可复现：同 seed + 同文本出同一个人，
        # 挑中了才好存下来当参考。
        self.seed = BREEZE_SEED if seed is None else seed
        self.timeout = timeout
        self._client = None                 # 连接复用，见 _cli()
        self._ref_bytes = None              # 参考音频只读一次盘
        # 服务端**单并发**：第二个请求在第一个还在流的时候打过去，直接 409 Conflict
        # （不是排队，是拒绝）。调用方是可以并发的——web 那边就提前给下一段起了任务，
        # 为的是省掉每段一个客户端→服务端的来回——所以这个约束由这里兜住：
        # 任务照样早创建，只是排在锁后面等，前一段一结束立刻发出去。
        # 锁只加在这个类里，OpenAI 那些能真并发的后端不受影响。
        self._lock = asyncio.Lock()
        # 服务端起服务时就把权重定死了，这里收下只为和别的后端对齐签名——
        # web/run.py 的 CONFIG 无论哪个 provider 都会传 model 进来。
        self.model = model

    def _cli(self):
        """复用同一个 client。一段回复是**逐句**发请求的（speak_stream 攒够一句就发），
        每句都新建连接的话，每次都要重走一遍 TCP 握手——服务在远端、又隔着 SSH 隧道
        时这一趟就是几十上百毫秒，直接听成卡顿。keep-alive 之后只有第一句付这个钱。
        """
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0))
        return self._client

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _raw(self, text, instruction=None):
        try:
            import httpx  # noqa: F401
        except ImportError as e:                # 跟着 openai 装的，正常都在
            raise ImportError("Breeze 后端要 httpx：pip install httpx") from e

        ins = instruction or self.instruction
        cfg = self.cfg_scale if self.cfg_scale is not None else (4 if ins else None)
        fields = {"text": text}
        if ins:
            fields["instruction"] = ins
        if cfg is not None:
            fields["cfg_scale"] = str(cfg)
        if self.ref_text:
            fields["ref_text"] = self.ref_text
        if self.seed is not None:
            fields["seed"] = str(self.seed)

        # 全部塞进 files 发 multipart：httpx 只在 files 非空时才发 multipart/form-data，
        # 光给 data= 会变成 urlencoded，而服务端那边示例是 curl -F。
        files = {k: (None, v) for k, v in fields.items()}
        if self.ref_audio:
            if self._ref_bytes is None:      # 每句话都去读一遍盘没必要
                from pathlib import Path
                ref = Path(self.ref_audio)
                self._ref_bytes = (ref.name, ref.read_bytes())
            files["ref_audio"] = self._ref_bytes

        url = f"{self.base_url}/v1/audio/speech"
        async with self._lock:                           # 单并发，见 __init__
            async with self._cli().stream("POST", url, files=files) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():   # 直接就是 24k 单声道 PCM16
                    yield chunk


#: provider 名 → 内置实现。``local`` 是 ``piper`` 的历史别名（TTS_BACKEND=local 一直
#: 是这个意思），两个都留着。
TTS_PROVIDERS = {
    "cosyvoice3": None,
    "openai": OpenAITTS,
    "local":  PiperTTS,
    "piper":  PiperTTS,
    "voxcpm": VoxCPMTTS,
    "breeze": BreezeTTS,
}


#: 按 (provider, config) 复用实例。piper / voxcpm 要加载模型（VoxCPM2 是 2B），
#: 每次新建等于重新加载一遍——而 demo 里每个 Memory Space 一个 VoiceMem 实例，
#: 不共享的话开三个空间就是三份模型。后端本身除了那个模型没有别的状态，可以共享。
_INSTANCES: dict = {}


def make_tts(provider: str | None = None, **cfg):
    """按 provider 名建一个内置后端；``provider`` 省略就跟 ``TTS_BACKEND`` 环境变量。

    ``cfg`` 逐个 provider 不同（openai 认 model/voice/instructions/api_key/base_url，
    piper 和 voxcpm 只认 model），传错了直接 TypeError——比静默忽略好找。

    同样的 (provider, cfg) 返回同一个实例。要各自独立的就直接构造类。
    """
    name = (provider or TTS_BACKEND).lower()
    cls = TTS_PROVIDERS.get(name)
    if name == "cosyvoice3":
        from voicemem.tts_cosyvoice import CosyVoice3TTS
        cls = CosyVoice3TTS
    if cls is None:
        raise ValueError(f"未知的 tts.provider={provider!r}；"
                         f"可选：{' / '.join(sorted(set(TTS_PROVIDERS)))}")
    key = (name, str(sorted(cfg.items())))
    if key not in _INSTANCES:
        _INSTANCES[key] = cls(**cfg)
    return _INSTANCES[key]


# ── 模块级入口（向后兼容）───────────────────────────────────────────────────────

def _tts_cfg(reply):
    seg = (reply or {}).get("tts") or {}
    return seg.get("provider"), (seg.get("config") or {})


async def tts_stream(text, reply=None, instruction=None):
    """从 ``reply.tts`` 那段配置解析后端并合成，吐 24kHz PCM16 流。

    注入进来的 TTS 走 ``vm.utils.get("tts").stream(text)``，不经过这里；这个函数
    是给只有一份 reply 配置、手上没有 VoiceMem 实例的调用方用的。
    """
    provider, cfg = _tts_cfg(reply)
    async for pcm in make_tts(provider, **cfg).stream(text, instruction):
        yield pcm


async def speak_stream(deltas, reply=None, on_delta=None, tts=None,
                       instruction=None):
    """文本增量流 → 语音流。合成跟生成**并行**：吐满一句就丢进队列，另一条协程
    取出来合成，边生成边出声。

    ``deltas``：异步迭代器（``vm.reply_stream(turn)`` 就是）。
    ``on_delta``：每收到一个文本增量回调一次（想边说边打字就传它）。
    ``tts``：合成用的对象（``vm.utils.get("tts")`` 或自己那个）；不给就按 ``reply``
    里的配置现解析一个。

    """
    queue: asyncio.Queue = asyncio.Queue()
    out: asyncio.Queue = asyncio.Queue()

    async def synth():
        while (seg := await queue.get()) is not None:
            gen = (tts.stream(seg, instruction) if tts is not None
                   else tts_stream(seg, reply, instruction))
            async for pcm in gen:
                await out.put(pcm)
        await out.put(None)

    worker = asyncio.create_task(synth())

    async def feed():
        buf, sent = "", 0
        try:
            async for d in deltas:
                if on_delta:
                    on_delta(d)
                buf += d
                if cut_point(buf, first=sent == 0):
                    await queue.put(buf.strip())
                    buf, sent = "", sent + 1
            if buf.strip():
                await queue.put(buf.strip())
        finally:
            await queue.put(None)               # Anche se la generazione va in errore, synth() deve terminare

    feeder = asyncio.create_task(feed())
    try:
        while (pcm := await out.get()) is not None:
            yield pcm
    finally:
        for t in (feeder, worker):
            if not t.done():
                t.cancel()
        await asyncio.gather(feeder, worker, return_exceptions=True)
