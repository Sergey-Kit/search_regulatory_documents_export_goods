"""Оффлайн-оценка конфигураций на dev по кэшу exp/cache."""
import json, sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_declarations, load_regulations
from src.corpus import build_docs
from src import preprocess
from src.retrieval import BM25, rrf, topk
from evaluate import metrics

ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "exp/cache"
decs = load_declarations(ROOT); regs = load_regulations(ROOT); docs = build_docs(regs)
ids = [d.regulation_id for d in docs]
labels = {l["idx"]: l["relevant"] for l in (json.loads(l) for l in open(ROOT / "data/dev_labels.jsonl"))}
q_lem = [preprocess.lemmas(d["G31_1"]) for d in decs]
bmB = BM25([preprocess.lemmas(d.text_for_bm25()) for d in docs])
bmC = BM25([preprocess.lemmas(d.body) for d in docs])
D = {k: np.load(C / f"doc_{k}.npy") for k in "ABC"}
Q = {"full": np.load(C / "q_full.npy"), "head": np.load(C / "q_head.npy"),
     "head150": np.load(C / "q_head150.npy"), "head400": np.load(C / "q_head400.npy")}
E = {k: np.load(C / f"exp_{k}.npy") for k in ["bm25", "dense", "rrf"]}
EXP = json.load(open(C / "tnved_exp.json"))
pool = {int(k): v for k, v in json.load(open(C / "pool.json")).items()}
RR = {k: {int(i): np.array(v) for i, v in json.load(open(C / f"rerank_{k}.json")).items()}
      for k in "ABChgij" if (C / f"rerank_{k}.json").exists()}


def sig(x): return 1 / (1 + np.exp(-x))


def run(cfg):
    res = []
    for i, rel in labels.items():
        bm = bmC if cfg.get("bm25_body_only") else bmB
        s_b = bm.scores(q_lem[i])
        lists = [(s_b, cfg["w_bm25"])]
        for qk in (["full", cfg.get("head", "head")] if cfg["q"] == "both" else [cfg["q"]]):
            lists.append((Q[qk][i] @ D[cfg["doc"]].T, cfg["w_dense"]))
        if cfg["w_tnved"] > 0:
            e = EXP[cfg["exp"]][i]
            lists.append((bm.scores(preprocess.lemmas(e)), cfg["w_tnved"]))
            lists.append((E[cfg["exp"]][i] @ D[cfg["doc"]].T, cfg["w_tnved"]))
        fused = rrf(lists, k=cfg.get("rrf_k", 60))
        if cfg["rerank"] is None:
            order = topk(fused, 10)
        else:
            cand = topk(fused, cfg["K"])
            lg = RR[cfg["rerank"]][i]; pmap = {j: n for n, j in enumerate(pool[i])}
            logit = np.array([lg[pmap[j]] if j in pmap else np.nan for j in cand])
            h = fused[cand]; h = (h - h.min()) / (h.max() - h.min() + 1e-9)
            score = np.where(np.isnan(logit), h - 1, sig(np.nan_to_num(logit)) + cfg["lam"] * h)
            order = cand[np.argsort(-score, kind="stable")][:10]
        res.append(metrics([ids[j] for j in order], rel))
    keys = ["nDCG@10", "MRR@10", "R@10", "R_exact@10", "P@1"]
    return {k: float(np.nanmean([r[k] for r in res])) for k in keys}


base = dict(q="full", doc="A", w_bm25=1.0, w_dense=1.0, w_tnved=0.5, exp="rrf", rerank="A", K=40, lam=0.1)
variants = {
    "hybrid C, no tnved": dict(base, w_tnved=0, rerank=None, doc="C"),
    "hybrid B head-q": dict(base, w_tnved=0, rerank=None, doc="B", q="head"),
    "hybrid C head-q": dict(base, w_tnved=0, rerank=None, doc="C", q="head"),
    "hybrid B both-q": dict(base, w_tnved=0, rerank=None, doc="B", q="both"),
    "hybrid C both-q": dict(base, w_tnved=0, rerank=None, doc="C", q="both"),
    "hybrid C both-q bm25 body": dict(base, w_tnved=0, rerank=None, doc="C", q="both", bm25_body_only=True),
    "hybrid C both-q rrf_k 20": dict(base, w_tnved=0, rerank=None, doc="C", q="both", rrf_k=20),
    "hybrid C both-q bm25x1.5": dict(base, w_tnved=0, rerank=None, doc="C", q="both", w_bm25=1.5),
    "hybrid C both-q dense x0.7": dict(base, w_tnved=0, rerank=None, doc="C", q="both", w_dense=0.7),
    "bm25 only": dict(base, w_dense=0, w_tnved=0, rerank=None),
    "dense A only": dict(base, w_bm25=0, w_tnved=0, rerank=None),
    "dense B only": dict(base, w_bm25=0, w_tnved=0, rerank=None, doc="B"),
    "dense C only": dict(base, w_bm25=0, w_tnved=0, rerank=None, doc="C"),
    "dense B head-query": dict(base, w_bm25=0, w_tnved=0, rerank=None, doc="B", q="head"),
    "hybrid A, no tnved": dict(base, w_tnved=0, rerank=None),
    "hybrid B, no tnved": dict(base, w_tnved=0, rerank=None, doc="B"),
    "hybrid B + tnved rrf 0.5": dict(base, rerank=None, doc="B"),
    "hybrid B + tnved rrf 0.3": dict(base, rerank=None, doc="B", w_tnved=0.3),
    "hybrid B + tnved dense-exp 0.5": dict(base, rerank=None, doc="B", exp="dense"),
    "hybrid B + tnved bm25-exp 0.5": dict(base, rerank=None, doc="B", exp="bm25"),
    "hybrid B (bm25 x2)": dict(base, rerank=None, doc="B", w_tnved=0, w_bm25=2.0),
    "hybridC both + rerank C": dict(base, w_tnved=0, doc="C", q="both", rerank="C"),
    "hybridC both + rerank C lam1": dict(base, w_tnved=0, doc="C", q="both", rerank="C", lam=1.0),
    "hybridC both + rerank C lam3": dict(base, w_tnved=0, doc="C", q="both", rerank="C", lam=3.0),
    "hybridC both + rerank h": dict(base, w_tnved=0, doc="C", q="both", rerank="h"),
    "hybridC both + rerank h lam1": dict(base, w_tnved=0, doc="C", q="both", rerank="h", lam=1.0),
    "hybridC both + rerank h lam3": dict(base, w_tnved=0, doc="C", q="both", rerank="h", lam=3.0),
    "hybridC both + rerank h K20 lam1": dict(base, w_tnved=0, doc="C", q="both", rerank="h", lam=1.0, K=20),
    "hybridC both + rerank B lam1": dict(base, w_tnved=0, doc="C", q="both", rerank="B", lam=1.0),
    "hybrid C both(150)": dict(base, w_tnved=0, rerank=None, doc="C", q="both", head="head150"),
    "hybrid C both(400)": dict(base, w_tnved=0, rerank=None, doc="C", q="both", head="head400"),
    "hybridC both + rerank i(150)": dict(base, w_tnved=0, doc="C", q="both", rerank="i"),
    "hybridC both + rerank j(400)": dict(base, w_tnved=0, doc="C", q="both", rerank="j"),
    "hybridA + rerank A (текущее)": dict(base),
    "hybridB + rerank A": dict(base, doc="B"),
    "hybridB + rerank B": dict(base, doc="B", rerank="B"),
    "hybridB notnved + rerank B": dict(base, doc="B", rerank="B", w_tnved=0),
    "hybridB + rerank B lam 0.3": dict(base, doc="B", rerank="B", lam=0.3),
    "hybridB + rerank B lam 1.0": dict(base, doc="B", rerank="B", lam=1.0),
    "hybridB + rerank B K20": dict(base, doc="B", rerank="B", K=20),
    "hybridB + rerank B K60": dict(base, doc="B", rerank="B", K=60),
}
print(f"{'вариант':36}" + "".join(f"{k:>12}" for k in ["nDCG@10", "MRR@10", "R@10", "R_exact@10", "P@1"]))
for name, cfg in variants.items():
    m = run(cfg)
    print(f"{name:36}" + "".join(f"{m[k]:12.3f}" for k in ["nDCG@10", "MRR@10", "R@10", "R_exact@10", "P@1"]))
