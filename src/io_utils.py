"""Чтение входных данных и запись/проверка predictions.csv."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["declaration_id", "rank", "regulation_id", "score"]
TOP_N = 10


def read_jsonl(path: str | Path) -> list[dict]:
    """Читает JSON Lines. Пустые строки пропускаются."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_input(root: Path, stem: str) -> Path:
    """В задании файлы названы *.jsonl, в комплекте — *.json; принимаем оба."""
    for name in (f"{stem}.jsonl", f"{stem}.json"):
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"не найден {stem}.jsonl / {stem}.json в {root}")


def load_declarations(root: Path) -> list[dict]:
    rows = read_jsonl(find_input(root, "declarations"))
    ids = [r["declaration_id"] for r in rows]
    assert len(ids) == len(set(ids)), "дубликаты declaration_id"
    return rows


def load_regulations(root: Path) -> list[dict]:
    rows = read_jsonl(find_input(root, "regulations"))
    ids = [r["regulation_id"] for r in rows]
    assert len(ids) == len(set(ids)), "дубликаты regulation_id"
    return rows


def validate_predictions(df: pd.DataFrame, declaration_ids, regulation_ids) -> None:
    """Проверяет формат по условиям задания; бросает AssertionError с описанием проблемы."""
    assert list(df.columns) == REQUIRED_COLUMNS, f"колонки должны быть {REQUIRED_COLUMNS}"
    decl = set(declaration_ids)
    regs = set(regulation_ids)
    assert len(df) == TOP_N * len(decl), f"ожидалось {TOP_N * len(decl)} строк, есть {len(df)}"
    assert set(df["declaration_id"]) == decl, "набор declaration_id не совпадает с входом"
    unknown = set(df["regulation_id"]) - regs
    assert not unknown, f"неизвестные regulation_id: {sorted(unknown)[:5]}"
    assert df["score"].map(lambda s: isinstance(s, (int, float)) and math.isfinite(s)).all(), "score не конечное число"
    for did, g in df.groupby("declaration_id", sort=False):
        assert sorted(g["rank"].tolist()) == list(range(1, TOP_N + 1)), f"{did}: ранги не 1..{TOP_N}"
        assert g["regulation_id"].is_unique, f"{did}: повторные regulation_id"
        s = g.sort_values("rank")["score"].to_numpy()
        assert all(s[i] >= s[i + 1] for i in range(len(s) - 1)), f"{did}: score не убывает по рангу"


def write_predictions(rows: list[dict], out_dir: Path, declaration_ids, regulation_ids) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    validate_predictions(df, declaration_ids, regulation_ids)
    path = out_dir / "predictions.csv"
    df.to_csv(path, index=False, float_format="%.6f")
    return path
