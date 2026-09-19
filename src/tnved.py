"""ТН ВЭД как семантический мост.

Кодов ТН ВЭД нет ни в декларациях, ни в НПА, поэтому справочник используется только для
переформулировки запроса: описание товара → ближайшие позиции ТН ВЭД → их нормализованные
названия («Оборудование холодильное или морозильное прочее»). Такие названия написаны тем
же «канцелярским» языком, что и контрольные списки, и гасят шум декларации.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(\d{4,10})\s+\|\s*(.*?)\s*$")
_NORM = re.compile(r"\[(.+)\]\s*$")
_DASHES = re.compile(r"^(?:[–-]\s*)+")


def parse_tnved(path: Path) -> list[dict]:
    """Листовые коды с нормализованным названием в квадратных скобках + заголовок 4-значной позиции."""
    entries: list[dict] = []
    heading4 = ""
    in_hier = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.startswith("ИЕРАРХИЯ ТН ВЭД"):
                in_hier = True
                continue
            if s in ("ПРИМЕЧАНИЯ", "ПОЯСНЕНИЯ", "ОБЩИЕ ПОЛОЖЕНИЯ"):
                in_hier = False
                continue
            if not in_hier:
                continue
            m = _LINE.match(s)
            if not m:
                continue
            code, text = m.group(1), m.group(2)
            if len(code) == 4:
                heading4 = _DASHES.sub("", text).rstrip(":").strip()
                continue
            n = _NORM.search(text)
            if not n:
                continue
            name = n.group(1).strip()
            entries.append({"code": code, "heading": heading4, "name": name})
    return entries


def entry_text(e: dict) -> str:
    """Текст позиции для индексации: заголовок группы + нормализованное название."""
    if e["heading"] and e["heading"].lower() not in e["name"].lower():
        return f"{e['heading']}: {e['name']}"
    return e["name"]


def load_or_build_index(raw_path: Path, cache_path: Path) -> list[dict]:
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return [json.loads(l) for l in f]
    entries = parse_tnved(raw_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return entries
