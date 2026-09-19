"""Оценка на ручной dev-разметке + анализ согласованности методов.

python evaluate.py [--debug out/debug_rankings.jsonl] [--labels data/dev_labels.jsonl]

debug_rankings.jsonl пишет run.py: для каждой декларации top-50 по bm25 / dense / hybrid
и итоговый порядок после реранкера (final). dev_labels.jsonl — ручная разметка:
{"declaration_id": ..., "relevant": {"NPA0151": 2, "NPA0007": 1}, "note": "..."}
(2 — точное соответствие, 1 — частичное/возможное).
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

METHODS = ["bm25", "dense", "hybrid", "final"]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def metrics(ranked: list[str], rel: dict[str, int], k: int = 10) -> dict[str, float]:
    top = ranked[:k]
    gains = [rel.get(r, 0) for r in top]
    ideal = sorted(rel.values(), reverse=True)[:k]
    ndcg = dcg(gains) / dcg(ideal) if ideal else 0.0
    hits = [i for i, r in enumerate(top) if rel.get(r, 0) > 0]
    mrr = 1.0 / (hits[0] + 1) if hits else 0.0
    recall = len(hits) / len(rel) if rel else 0.0
    exact = [r for r, g in rel.items() if g == 2]
    r_exact = (sum(1 for r in top if r in exact) / len(exact)) if exact else float("nan")
    return {"nDCG@10": ndcg, "MRR@10": mrr, "R@10": recall, "R_exact@10": r_exact, "P@1": float(gains[0] == 2) if gains else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", default="out/debug_rankings.jsonl")
    ap.add_argument("--labels", default="data/dev_labels.jsonl")
    a = ap.parse_args()
    dbg = {r["declaration_id"]: r for r in read_jsonl(Path(a.debug))}
    labels = [l for l in read_jsonl(Path(a.labels))] if Path(a.labels).exists() else []

    if labels:
        print(f"dev-разметка: {len(labels)} деклараций, "
              f"{sum(len(l['relevant']) for l in labels)} релевантных пар\n")
        header = f"{'метод':8}" + "".join(f"{m:>12}" for m in ["nDCG@10", "MRR@10", "R@10", "R_exact@10", "P@1"])
        print(header)
        for m in METHODS:
            agg: dict[str, list[float]] = {}
            for l in labels:
                if l["declaration_id"] not in dbg:
                    continue
                res = metrics(dbg[l["declaration_id"]][m], l["relevant"])
                for k, v in res.items():
                    if not math.isnan(v):
                        agg.setdefault(k, []).append(v)
            print(f"{m:8}" + "".join(f"{sum(agg[k]) / len(agg[k]):12.3f}" for k in ["nDCG@10", "MRR@10", "R@10", "R_exact@10", "P@1"]))
        print()
        # пропуски: релевантные позиции, не попавшие в top-10 финального порядка
        misses = []
        for l in labels:
            fin = dbg[l["declaration_id"]]["final"]
            for r, g in l["relevant"].items():
                if g == 2 and r not in fin[:10]:
                    pos = fin.index(r) + 1 if r in fin else None
                    misses.append((l["declaration_id"], r, pos))
        print(f"точных соответствий вне top-10 (final): {len(misses)}")
        for did, r, pos in misses:
            print(f"  {did} {r} позиция в final: {pos}")
        print()

    # согласованность методов по всем декларациям: |top10 ∩ top10| / 10
    print("среднее пересечение top-10 между методами (все декларации):")
    for a_, b_ in combinations(METHODS, 2):
        ov = [len(set(r[a_][:10]) & set(r[b_][:10])) / 10 for r in dbg.values()]
        print(f"  {a_:7} ~ {b_:7}: {sum(ov) / len(ov):.2f}")


if __name__ == "__main__":
    main()
