"""Стенд экспериментов: считает и кэширует всё дорогое для dev-выборки (эмбеддинги, логиты реранкера).

python exp/build_cache.py   → exp/cache/*.npy, exp/cache/meta.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_declarations, load_regulations
from src.corpus import build_docs
from src import preprocess, tnved
from src.retrieval import BM25, DenseEncoder, rrf, topk
from src.rerank import Reranker

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "exp" / "cache"; C.mkdir(exist_ok=True)
decs = load_declarations(ROOT); regs = load_regulations(ROOT); docs = build_docs(regs)
labels = [json.loads(l) for l in open(ROOT / "data/dev_labels.jsonl")]
dev_idx = [l["idx"] for l in labels]

doc_variants = {
    "A": [d.text_for_model() for d in docs],                       # заголовок + текст + ссылки
    "B": [d.text_for_bm25() for d in docs],                        # текст + ссылки
    "C": [d.body for d in docs],                                   # только текст
}
q_full = [preprocess.normalize(d["G31_1"]) for d in decs]
q_head = [q[:250] for q in q_full]
q_lem = [preprocess.lemmas(d["G31_1"]) for d in decs]

enc = DenseEncoder(str(ROOT / "models/user-bge-m3"), "cuda")
t = time.perf_counter()
for k, texts in doc_variants.items():
    np.save(C / f"doc_{k}.npy", enc.encode(texts))
np.save(C / "q_full.npy", enc.encode(q_full)); np.save(C / "q_head.npy", enc.encode(q_head))
print("dense", time.perf_counter() - t)

# ТН ВЭД-расширения (с исправленной токенизацией)
entries = tnved.load_or_build_index(ROOT / "tnved_knowledge.txt", ROOT / "data/tnved_index.jsonl")
texts = [tnved.entry_text(e) for e in entries]
tn_bm = BM25([preprocess.lemmas(x) for x in texts])
tn_emb = np.load(ROOT / "data/tnved_emb.npy").astype(np.float32)
qe = np.load(C / "q_full.npy")
exp_bm25, exp_dense, exp_rrf = [], [], []
for i in range(len(decs)):
    sb = tn_bm.scores(q_lem[i]); sd = qe[i] @ tn_emb.T
    def names(idx):
        out = []
        for j in idx:
            n = entries[j]["name"]
            if n not in out: out.append(n)
        return "; ".join(out)
    exp_bm25.append(names(topk(sb, 5))); exp_dense.append(names(topk(sd, 5)))
    exp_rrf.append(names(topk(rrf([(sb, 1.0), (sd, 1.0)]), 5)))
json.dump({"bm25": exp_bm25, "dense": exp_dense, "rrf": exp_rrf}, open(C / "tnved_exp.json", "w"), ensure_ascii=False)
for k, v in [("bm25", exp_bm25), ("dense", exp_dense), ("rrf", exp_rrf)]:
    np.save(C / f"exp_{k}.npy", enc.encode(v))
enc.close()
print("tnved done", time.perf_counter() - t)

# пул кандидатов для dev: объединение top-40 по bm25 / dense(A) / dense(B)
bm = BM25([preprocess.lemmas(x) for x in doc_variants["B"]])
dA, dB = np.load(C / "doc_A.npy"), np.load(C / "doc_B.npy")
pool = {}
for i in dev_idx:
    s = set(topk(bm.scores(q_lem[i]), 40)) | set(topk(qe[i] @ dA.T, 40)) | set(topk(qe[i] @ dB.T, 40))
    s |= set(topk(np.load(C / "exp_rrf.npy")[i] @ dA.T, 20))
    pool[i] = sorted(s)
json.dump({str(k): [int(x) for x in v] for k, v in pool.items()}, open(C / "pool.json", "w"))
print("pool sizes", sum(len(v) for v in pool.values()))

rr = Reranker(str(ROOT / "models/bge-reranker-v2-m3"), "cuda")
for k in ["A", "B"]:
    t = time.perf_counter()
    logits = {}
    for i in dev_idx:
        logits[str(i)] = rr.score(q_full[i], [doc_variants[k][j] for j in pool[i]]).tolist()
    json.dump(logits, open(C / f"rerank_{k}.json", "w"))
    print("rerank", k, time.perf_counter() - t)
