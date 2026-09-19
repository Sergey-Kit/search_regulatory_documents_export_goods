"""Нормализация текста деклараций и НПА.

Две формы текста:
* ``normalize`` — «мягкая»: нижний регистр, ё→е, удаление маскированных номеров и
  склейка разорванных слов. Подаётся в нейросетевые модели.
* ``lemmas`` — токены-леммы (pymorphy3) для BM25.
"""
from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3

_MORPH = pymorphy3.MorphAnalyzer()

# «5XXXXX7», «1XXXXXГ.» — маскированные серийные номера/даты; несут только шум.
_MASKED = re.compile(r"\S*[xхXХ]{3,}\S*")
# Токен: буквы/цифры, внутри допускаем дефис и точку («ту-154», «1.92»).
_TOKEN = re.compile(r"[a-zа-я0-9]+(?:[-.][a-zа-я0-9]+)*")
_STOP = {
    "и", "в", "во", "на", "с", "со", "для", "по", "из", "от", "до", "или", "не", "а", "о", "об",
    "при", "к", "у", "за", "то", "же", "как", "что", "это", "но", "их", "его", "ее", "оно",
    "также", "либо", "либо", "более", "менее", "чем", "прочие", "прочий", "прочее",
    "the", "of", "and", "for", "with", "to", "in", "on", "or",
}


def _is_word(tok: str) -> bool:
    return tok.isalpha() and _MORPH.word_is_known(tok)


def merge_split_words(tokens: list[str]) -> list[str]:
    """Склеивает пары «т вердотельных» → «твердотельных», «каро тажа» → «каротажа».

    Условие: оба куска кириллические, хотя бы один неизвестен словарю, а склейка известна.
    Однобуквенный известный кусок («т», «с») тоже разрешаем склеивать, если сосед неизвестен.
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        a = tokens[i]
        if i + 1 < len(tokens):
            b = tokens[i + 1]
            if (
                re.fullmatch(r"[а-я]+", a) and re.fullmatch(r"[а-я]+", b)
                and (not _is_word(a) or not _is_word(b) or len(a) == 1 or len(b) == 1)
                and len(a) + len(b) >= 5
                and _is_word(a + b)
                and not (_is_word(a) and _is_word(b) and len(a) > 1 and len(b) > 1)
            ):
                out.append(a + b)
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def normalize(text: str) -> str:
    """Мягкая нормализация; сохраняет пунктуацию и числа для нейросетевых моделей."""
    text = text.lower().replace("ё", "е")
    text = _MASKED.sub(" ", text)
    text = text.replace("<", " ").replace(">", " ")
    # Склейка разорванных слов делается на уровне слов, пунктуацию сохраняем.
    parts = re.split(r"(\s+)", text)
    words = [p for p in parts if p and not p.isspace()]
    merged = merge_split_words([w for w in words])
    if len(merged) != len(words):
        text = " ".join(merged)
    text = re.sub(r"\s+", " ", text).strip(" :;,")
    return text


@lru_cache(maxsize=200_000)
def lemma(tok: str) -> str:
    if tok.isdigit() or not tok.isalpha():
        return tok
    return _MORPH.parse(tok)[0].normal_form.replace("ё", "е")


def lemmas(text: str) -> list[str]:
    """Токены-леммы для BM25 (со склейкой разорванных слов, без стоп-слов)."""
    text = normalize(text)
    toks = _TOKEN.findall(text)
    toks = merge_split_words(toks)
    return [lemma(t) for t in toks if t not in _STOP and len(t) > 1]
