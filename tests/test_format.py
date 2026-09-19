import math
from pathlib import Path

import pandas as pd
import pytest

from src.io_utils import TOP_N, load_declarations, load_regulations, validate_predictions

ROOT = Path(__file__).resolve().parents[1]


def _good(decls, regs):
    rows = []
    for d in decls:
        for r in range(1, TOP_N + 1):
            rows.append({"declaration_id": d, "rank": r, "regulation_id": regs[r - 1], "score": 1.0 - r * 0.01})
    return pd.DataFrame(rows)


def test_valid_passes():
    decls, regs = ["d1", "d2"], [f"r{i}" for i in range(20)]
    validate_predictions(_good(decls, regs), decls, regs)


@pytest.mark.parametrize("mutate,msg", [
    (lambda df: df.iloc[:-1], "строк"),
    (lambda df: df.assign(regulation_id=df["regulation_id"].replace({"r1": "r0"})), "повторные"),
    (lambda df: df.assign(rank=df["rank"].replace({2: 1})), "ранги"),
    (lambda df: df.assign(regulation_id=df["regulation_id"].replace({"r1": "zzz"})), "неизвестные"),
    (lambda df: df.assign(score=[math.nan] + [0.5] * (len(df) - 1)), "конечное"),
    (lambda df: df.assign(score=df["rank"].astype(float)), "убывает"),
])
def test_invalid_fails(mutate, msg):
    decls, regs = ["d1", "d2"], [f"r{i}" for i in range(20)]
    with pytest.raises(AssertionError, match=msg):
        validate_predictions(mutate(_good(decls, regs)), decls, regs)


@pytest.mark.skipif(not (ROOT / "out" / "predictions.csv").exists(), reason="predictions.csv ещё не получен")
def test_real_predictions_valid():
    df = pd.read_csv(ROOT / "out" / "predictions.csv")
    decls = [d["declaration_id"] for d in load_declarations(ROOT)]
    regs = [r["regulation_id"] for r in load_regulations(ROOT)]
    validate_predictions(df, decls, regs)
