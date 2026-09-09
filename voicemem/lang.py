"""Language support for VoiceMem — Italian and English only.

Lingue supportate: ``("it", "en")``. Default: ``"it"``.
Il cinese (zh) non è più una lingua selezionabile in questo fork.

La lingua della Memory Space è la sorgente di verità per ASR, prompt, memoria, risposta e UI.
"""

from __future__ import annotations

import os

#: 环境变量名。也可以 VoiceMem(memory_language="it")、demo 的 --lang。
ENV = "VOICEMEM_MEMORY_LANGUAGE"
SUPPORTED = ("it", "en")
DEFAULT = "it"

_override: str | None = None


def _check(value: str) -> str:
    v = (value or "").strip().lower()
    if v not in SUPPORTED:
        raise ValueError(f"memory_language 只能是 {' / '.join(SUPPORTED)}，收到 {value!r}")
    return v


def _set(lang: str) -> None:
    global _override
    _override = lang


def set_memory_language(value: str | None) -> None:
    """进程级设置。None / "" 表示清掉覆盖，回到 env / 默认。"""
    global _override
    _override = _check(value) if (value or "").strip() else None


def memory_language() -> str:
    if _override:
        return _override
    env = (os.environ.get(ENV, "") or "").strip()
    return _check(env) if env else DEFAULT


def resolve_for_space(memory_root, explicit: str | None = None) -> str:
    """把这个实例的语言定下来，并落到它对应的**空间**上。

    ``VoiceMem(memory_language=...)`` 以前只写进程级 override，于是：先建一个
    it 实例、再建一个不传参数的实例，后者会继承 it——文档写的默认 en 变成了
    取决于构造顺序（issue #9）。根子是同一个概念存了两个地方：demo 那边从空间
    json 读，库这边只改全局。

    这里统一到空间上：

        显式给了 → 写进这个空间的 json，并生效
        没给     → 读这个空间自己的记录；空间没记录再回落 env / 默认，
                   并把结果写回去（建的时候定一次，之后不再变）

    两个实例各读各的空间，构造顺序不再影响任何东西。
    """
    import json as _json
    from voicemem.utils.common import space as _space
    try:
        path = _space.json_path(memory_root)
    except Exception:
        # 拿不到空间目录（极少数纯内存用法）：退回原来的全局行为
        set_memory_language(explicit)
        return memory_language()

    stored = ""
    try:
        if path.exists():
            stored = (_json.loads(path.read_text(encoding="utf-8"))
                      .get("space", {}).get("language", "") or "")
    except Exception:
        stored = ""

    if explicit:
        lang = _check(explicit)
    elif stored:
        lang = _check(stored)
    else:
        env = (os.environ.get(ENV, "") or "").strip()
        lang = _check(env) if env else DEFAULT

    if lang != stored:
        try:
            doc = (_json.loads(path.read_text(encoding="utf-8"))
                   if path.exists() else {})
            doc.setdefault("space", {})["language"] = lang
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        except Exception as e:
            print(f"[lang] 写空间语言失败（不影响使用）：{e}", flush=True)

    _set(lang)
    return lang


def is_it() -> bool:
    return memory_language() == "it"


def is_en() -> bool:
    return memory_language() == "en"


def language_name() -> str:
    """Restituisce il nome leggibile della lingua corrente (per UI)."""
    return "Italiano" if is_it() else "English"


def label_rule() -> str:
    """拼进 prompt 的一句语言要求。所有存进记忆的自由文本都该带上它。"""
    lang = "Italiano" if is_it() else "English"
    return (f"Write every label in {lang}, whatever language the speaker used. "
            f"Do not mix in any other language.")


#: 8 个规范情绪（内部值一律中文，见 anchor_router._CANONICAL_EMOTIONS）→ 展示词。
#:
#: 内部值不能翻译：右脑的锚点匹配、配额、脑图聚类都按它做键。但**存进记忆、给
#: 用户看的那一份**要跟库语言一致，否则英文库里会冒出「开心」「平静」。
#: 这里挑的英文词都在 anchor_router._EMOTION_KEYWORDS_EN 里，所以英文标签再被
#: 读回来时能正确归一回同一个内部值，不会丢。
_EMOTION_IT = {
    "anxious": "ansioso", "sad": "triste", "wronged": "ingiustizzato", "lonely": "solo",
    "conflicted": "indeciso", "calm": "calmo", "happy": "felice", "tired": "stanco",
}
_EMOTION_EN = {
    "ansioso": "anxious", "triste": "sad", "ingiustizzato": "wronged", "solo": "lonely",
    "indeciso": "conflicted", "calmo": "calm", "felice": "happy", "stanco": "tired",
}


def display_emotion(canonical: str) -> str:
    """规范情绪 → 当前库语言下的写法。不认识的原样返回。"""
    if is_it():
        return _EMOTION_IT.get(canonical, canonical)
    return _EMOTION_EN.get(canonical, canonical)
