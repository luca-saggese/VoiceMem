"""Espande le parole temporali relative nelle domande in date assolute, poi le usa per il retrieval.

Perché serve: l'estrazione raggrupperà "visita medica mercoledì prossimo alle 15:00" in "Jiaqi avrà una visita medica il 26 agosto 2026 (mercoledì) alle 15:00" — nel database sono memorizzate date assolute. Ma l'utente chiede "quali sono i miei impegni della prossima settimana", e questa frase
**non contiene alcuna data assoluta**, i vettori non corrispondono, nei test reali nessuna delle tre programmazioni della domenica successiva viene recuperata::

    "Quali sono i miei impegni della prossima settimana"   Hit programmazione 0/3
    "Cosa devo fare il 26 agosto"    Hit programmazione 3/3

La differenza è solo nel modo di formulare. Quindi prima del retrieval espandi "prossima settimana" nelle date di quei sette giorni, aggiungendole dopo la domanda —
con parole come "26 agosto" presenti nel vettore, si può raggiungere quel record nel database.

Modifica solo **il testo usato per il retrieval**, non le parole dell'utente e non lo scrivi nella memoria.

    expand_relative_dates("Quali sono i miei impegni della prossima settimana?")
    → "Quali sono i miei impegni della prossima settimana? (24 ago 2026 25 ago 2026 … 30 ago 2026)"

Se non riconosce parole temporali relative, restituisce invariato senza spendere nulla (solo regex, nessun modello).
"""
from __future__ import annotations

import re
from datetime import date, timedelta

#: 相对时间词 → (起始偏移, 天数)。偏移是相对"今天"的天数。
#: 周相关的偏移在 _resolve 里按当天星期几现算，这里用 None 占位。
_SPANS: dict[str, tuple[int | None, int]] = {
    "前天":     (-2, 1),
    "昨天":     (-1, 1),
    "今天":     (0, 1),
    "今日":     (0, 1),
    "明天":     (1, 1),
    "明日":     (1, 1),
    "后天":     (2, 1),
    "大后天":   (3, 1),
    "这几天":   (0, 3),
    "最近几天": (-3, 4),
    "接下来几天": (0, 4),
    "未来几天": (0, 4),
    # 周：偏移按当天星期几算
    "上周":     (None, 7),
    "上个星期": (None, 7),
    "这周":     (None, 7),
    "本周":     (None, 7),
    "这个星期": (None, 7),
    "下周":     (None, 7),
    "下个星期": (None, 7),
    "下星期":   (None, 7),
}

#: 周相关的词 → 相对"本周一"的周偏移
_WEEK_OFFSET = {
    "上周": -1, "上个星期": -1,
    "这周": 0, "本周": 0, "这个星期": 0,
    "下周": 1, "下个星期": 1, "下星期": 1,
}

#: 长的词要先匹配，否则"下个星期"会先被"下周"之外的短词切碎。
_WORDS = sorted(_SPANS, key=len, reverse=True)
_RE = re.compile("|".join(re.escape(w) for w in _WORDS))

#: 一次最多展开几个日期。问"最近三个月"这种展开出来上百个日期，
#: 会把问句本身的语义冲淡，反而检索更差。
_MAX_DAYS = 8


def _resolve(word: str, today: date) -> list[date]:
    """一个相对时间词覆盖哪几天。"""
    if word in _WEEK_OFFSET:
        monday = today - timedelta(days=today.weekday())      # 本周一
        start = monday + timedelta(weeks=_WEEK_OFFSET[word])
        return [start + timedelta(days=i) for i in range(7)]
    offset, days = _SPANS[word]
    start = today + timedelta(days=offset or 0)
    return [start + timedelta(days=i) for i in range(days)]


def expand_relative_dates(query: str, today: date | None = None) -> str:
    """在问句后面补上它涉及的绝对日期。没有相对时间词就原样返回。"""
    if not query:
        return query
    words = _RE.findall(query)
    if not words:
        return query

    today = today or date.today()
    days: list[date] = []
    for word in words:
        for d in _resolve(word, today):
            if d not in days:
                days.append(d)
    if not days or len(days) > _MAX_DAYS:
        return query

    days.sort()
    # 写成"2026年8月26日"这种格式，跟抽取归一后的写法对齐——格式不一样就白展开了。
    stamps = " ".join(f"{d.year}年{d.month}月{d.day}日" for d in days)
    return f"{query}（{stamps}）"
