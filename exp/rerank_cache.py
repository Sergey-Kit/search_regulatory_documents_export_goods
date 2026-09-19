import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_declarations, load_regulations
from src.corpus import build_docs
from src import preprocess
from src.rerank import Reranker
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "exp/cache"
decs = load_declarations(ROOT); docs = build_docs(load_regulations(ROOT))
pool = {int(k): v for k, v in json.load(open(C / "pool.json")).items()}
q_full = [preprocess.normalize(d["G31_1"]) for d in decs]
q_head = [q[:250] for q in q_full]
# вариант = (запрос, документ): A/B — полный запрос; h — «голова» запроса; C — только текст позиции
variants = {"A": (q_full, [d.text_for_model() for d in docs]), "B": (q_full, [d.text_for_bm25() for d in docs]),
            "C": (q_full, [d.body for d in docs]), "h": (q_head, [d.text_for_bm25() for d in docs]),
            "g": (q_head, [d.body for d in docs]),
            "i": ([q[:150] for q in q_full], [d.text_for_bm25() for d in docs]),
            "j": ([q[:400] for q in q_full], [d.text_for_bm25() for d in docs]),
            "n": ([preprocess.product_name(d["G31_1"]) for d in decs], [d.text_for_bm25() for d in docs]),
            "m": ([preprocess.product_name(d["G31_1"]) + ". " + q[:250] for d, q in zip(decs, q_full)], [d.text_for_bm25() for d in docs])}
rr = Reranker(str(ROOT / "models/bge-reranker-v2-m3"), "cuda", batch_size=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
for k in sys.argv[1]:
    t = time.perf_counter(); logits = {}; n = 0
    for i, cand in pool.items():
        qs, dv = variants[k]
        logits[str(i)] = rr.score(qs[i], [dv[j] for j in cand]).tolist(); n += len(cand)
        print(k, i, n, f"{time.perf_counter() - t:.0f}s VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)
    json.dump(logits, open(C / f"rerank_{k}.json", "w"))
