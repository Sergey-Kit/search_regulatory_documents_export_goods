import sys; from pathlib import Path
sys.argv = ['x']
src = open(Path(__file__).resolve().parent / 'eval_variants.py').read().split('base = dict')[0]
exec(src)
base = dict(q="both", doc="C", w_bm25=1.0, w_dense=1.0, w_tnved=0, exp="rrf", rerank="h", K=40, lam=0.1)
variants = {"итог (K20)": dict(base, K=20),
    "RoSBERTa only (full+head), без реранкера": dict(base, rerank=None, w_bm25=0, w_dense=0, w_ros=1.0),
    "гибрид без реранкера": dict(base, rerank=None),
    "гибрид + RoSBERTa 0.5, без реранкера": dict(base, rerank=None, w_ros=0.5),
    "гибрид + RoSBERTa 1.0, без реранкера": dict(base, rerank=None, w_ros=1.0),
    "гибрид + RoSBERTa 1.0 (doc B), без реранкера": dict(base, rerank=None, w_ros=1.0, ros_doc="B"),
    "гибрид + RoSBERTa 0.5 + реранкер": dict(base, K=20, w_ros=0.5),
    "гибрид + RoSBERTa 1.0 + реранкер": dict(base, K=20, w_ros=1.0),
    "гибрид + RoSBERTa 1.0 + реранкер K30": dict(base, K=30, w_ros=1.0),
}
variants["реранкер: запрос = название"] = dict(base, K=20, rerank="n")
variants["реранкер: название + голова"] = dict(base, K=20, rerank="m")
variants["+name dense 1.0, без реранкера"] = dict(base, rerank=None, w_name=1.0)
variants["без name, без реранкера"] = dict(base, rerank=None)
for name, cfg in variants.items():
    m = run(cfg); print(f"{name:20}", {k: round(v, 3) for k, v in m.items()})
