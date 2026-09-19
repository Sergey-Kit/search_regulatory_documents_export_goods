"""Сборка текстов документов НПА для поиска.

Документ = название контрольного списка + текст позиции + (для указа 1661) тексты
пунктов, на которые позиция ссылается («…указанных в пункте 6.1.4.4.2»).
Ссылки разрешаются только внутри того же указа и только если целевой пункт есть в
выборке — у остальных указов номера пунктов в тексте отсутствуют.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Человеческие названия списков — контекст для dense-модели и реранкера.
DECREE_TITLES = {
    "1661": "Список товаров и технологий двойного назначения, которые могут быть использованы "
            "при создании вооружений и военной техники (экспортный контроль)",
    "202": "Список ядерных материалов, оборудования, специальных неядерных материалов и "
           "технологий, подпадающих под экспортный контроль",
    "36": "Список оборудования и материалов двойного назначения и технологий, "
          "применяемых в ядерных целях",
    "1005": "Список оборудования, материалов и технологий, которые могут быть использованы "
            "при создании ракетного оружия",
    "1082": "Список химикатов, оборудования и технологий, которые могут быть использованы "
            "при создании химического оружия",
    "1083": "Список микроорганизмов, токсинов, оборудования и технологий, которые могут "
            "быть использованы при создании бактериологического (биологического) и "
            "токсинного оружия",
    "КЕЭК30": "Единый перечень товаров, к которым применяются меры нетарифного регулирования "
              "ЕАЭС (Решение Коллегии ЕЭК № 30)",
}

_ITEM_PREFIX = re.compile(r"^(?:раздел\s+(\d+),\s*)?(\d+(?:\.\d+)+)\.?\s*", re.I)
# «в пункте 6.1.6.1, 6.1.6.2 или 6.1.6.3», «подпункте "а" пункта 6.1.4.4.2», «позициях 9.1.3 или 9.1.5»
_REF = re.compile(
    r"(?:пункт|подпункт|позици|п\.)[а-я]*\s*(?:\"[а-я]\"\s*)?(?:пункта\s+)?"
    r"(\d+(?:\.\d+){1,}(?:\.?\s*(?:,|или|и|;)\s*\d+(?:\.\d+){1,})*)",
    re.I,
)
_NUM = re.compile(r"\d+(?:\.\d+)+")
_REF_MAX_CHARS = 300


@dataclass
class Doc:
    regulation_id: str
    decree: str
    item_no: str | None
    body: str                       # исходный текст без префикса «Раздел N, x.y.z.»
    refs: list[str] = field(default_factory=list)
    ref_texts: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return DECREE_TITLES.get(self.decree, f"Указ/решение № {self.decree}")

    def text_for_model(self) -> str:
        """Текст для dense-поиска и реранкера: заголовок списка + позиция + ссылки."""
        parts = [self.title + ".", self.body]
        if self.ref_texts:
            parts.append("Связанные позиции: " + " | ".join(self.ref_texts))
        return " ".join(parts)

    def text_for_bm25(self) -> str:
        """Для BM25 заголовок списка не добавляем: он общий для сотен позиций и лишь размывает веса."""
        parts = [self.body]
        if self.ref_texts:
            parts.append(" ".join(self.ref_texts))
        return " ".join(parts)


def split_prefix(npa: str) -> tuple[str | None, str]:
    m = _ITEM_PREFIX.match(npa)
    if not m:
        return None, npa
    return m.group(2), npa[m.end():]


def find_refs(text: str) -> list[str]:
    seen: list[str] = []
    for m in _REF.finditer(text):
        for no in _NUM.findall(m.group(1)):
            if no not in seen:
                seen.append(no)
    return seen


def build_docs(regulations: list[dict]) -> list[Doc]:
    docs = []
    for r in regulations:
        item_no, body = split_prefix(r["npa"])
        docs.append(Doc(r["regulation_id"], str(r["decree_number"]), item_no, body.strip()))
    index = {(d.decree, d.item_no): d for d in docs if d.item_no}
    for d in docs:
        d.refs = [no for no in find_refs(d.body) if no != d.item_no]
        for no in d.refs:
            tgt = index.get((d.decree, no))
            if tgt is not None:
                d.ref_texts.append(tgt.body[:_REF_MAX_CHARS])
    return docs
