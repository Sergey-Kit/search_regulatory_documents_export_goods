"""Досчитать логиты реранкера для кандидатов новых гибридов, которых нет в пуле."""
import json, sys
from pathlib import Path
import numpy as np
sys.argv = ['x']
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "exp/cache"
src = open(ROOT / 'exp/eval_variants.py').read().split('base = dict')[0]
exec(src)
from src.rerank import Reranker
base = dict(q="both", doc="D", w_bm25=1.0, w_dense=1.0, w_tnved=0, exp="rrf", rerank=None, K=40, lam=0.1,
            w_ros=1.0, ros_doc="D", bm25_cat=True, w_char=1.0, w_char_head=1.0)
# кандидаты: top-40 нового гибрида для каждой dev-декларации
need = {}
for i in labels:
    bm = bmD; s_b = bm.scores(q_lem[i]); lists = [(s_b, 1.0)]
    for qk in ["full", "head"]:
        lists.append((Q[qk][i] @ D["D"].T, 1.0)); lists.append((ROS["q_" + qk][i] @ ROS["doc_D"].T, 1.0))
    lists += [(CHAR[i], 1.0), (CHAR_H[i], 1.0)]
    need[i] = [int(j) for j in topk(rrf(lists, k=60), 40)]
pool = {int(k): v for k, v in json.load(open(C / "pool.json")).items()}
extra = {i: [j for j in need[i] if j not in pool[i]] for i in need}
print("недостающих пар:", sum(len(v) for v in extra.values()))
q_head = [preprocess.normalize(d["G31_1"])[:250] for d in decs]
rr = Reranker(str(ROOT / "models/bge-reranker-v2-m3"), "cuda")
variants = {"h": [d.text_for_bm25() for d in docs],
            "hD": [d.text_with_category() + (" Связанные позиции: " + " | ".join(d.ref_texts) if d.ref_texts else "") for d in docs]}
for k, texts in variants.items():
    f = C / f"rerank_{k}.json"
    cur = {int(a): list(b) for a, b in json.load(open(f)).items()} if f.exists() else {}
    for i in need:
        full = pool[i] + extra[i]
        if k == "hD":  # для нового варианта считаем весь пул
            cur[i] = rr.score(q_head[i], [texts[j] for j in full]).tolist()
        elif extra[i]:
            cur[i] = cur[i] + rr.score(q_head[i], [texts[j] for j in extra[i]]).tolist()
    json.dump({str(a): b for a, b in cur.items()}, open(f, "w"))
    print(k, "готово")
for i in need:
    pool[i] = pool[i] + extra[i]
json.dump({str(k): v for k, v in pool.items()}, open(C / "pool.json", "w"))
