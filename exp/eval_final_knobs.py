import sys; from pathlib import Path
sys.argv = ['x']
src = open(Path(__file__).resolve().parent / 'eval_variants.py').read().split('base = dict')[0]
exec(src)
base = dict(q="both", doc="C", w_bm25=1.0, w_dense=1.0, w_tnved=0, exp="rrf", rerank="h", K=40, lam=0.1)
base = dict(base, w_ros=1.0)   # текущий итог: 5 списков
new = dict(base, w_char=1.0, w_char_head=1.0, doc="D", ros_doc="D", bm25_cat=True)
variants = {
    "старый итог + реранкер h": dict(base, K=20),
    "новый гибрид, без реранкера": dict(new, rerank=None),
    "новый + реранкер h, K20": dict(new, K=20, rerank="h"),
    "новый + реранкер h, K30": dict(new, K=30, rerank="h"),
    "новый + реранкер h, K40": dict(new, K=40, rerank="h"),
    "новый + реранкер hD (с категорией), K20": dict(new, K=20, rerank="hD"),
    "новый + реранкер hD, K40": dict(new, K=40, rerank="hD"),
    "новый + реранкер h, lam 0.3": dict(new, K=20, rerank="h", lam=0.3),
    "новый + реранкер h, lam 1.0": dict(new, K=20, rerank="h", lam=1.0),
    "новый + реранкер hD, lam 0.3": dict(new, K=20, rerank="hD", lam=0.3),
    "новый + реранкер hD, lam 1.0": dict(new, K=20, rerank="hD", lam=1.0),
}
for name, cfg in variants.items():
    m = run(cfg); print(f"{name:20}", {k: round(v, 3) for k, v in m.items()})
