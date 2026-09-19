import sys; from pathlib import Path
sys.argv = ['x']
src = open(Path(__file__).resolve().parent / 'eval_variants.py').read().split('base = dict')[0]
exec(src)
base = dict(q="both", doc="C", w_bm25=1.0, w_dense=1.0, w_tnved=0, exp="rrf", rerank="h", K=40, lam=0.1)
for name, cfg in {"K20": dict(base, K=20), "K30": dict(base, K=30), "K40": base, "K60 (пул неполный)": dict(base, K=60),
                  "lam0": dict(base, lam=0.0), "lam0.3": dict(base, lam=0.3)}.items():
    m = run(cfg); print(f"{name:20}", {k: round(v, 3) for k, v in m.items()})
