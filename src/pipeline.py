"""Оркестрация: данные → BM25/dense → RRF → реранкер → predictions.csv."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import corpus, preprocess, tnved
from .io_utils import TOP_N, load_declarations, load_regulations, write_predictions
from .retrieval import BM25, DenseEncoder, rrf, topk
from .rerank import Reranker, rerank_all

log = logging.getLogger("pipeline")


@dataclass
class Config:
    root: Path
    out_dir: Path
    models_dir: Path
    device: str = "cpu"
    rerank_k: int = 40
    time_budget_s: float = 1200.0
    use_tnved: bool = True
    tnved_top: int = 5
    w_bm25: float = 1.0
    w_dense: float = 1.0
    w_tnved: float = 0.5           # вес каждого из двух ТН ВЭД-списков в RRF
    rrf_k: int = 60
    lam: float = 0.1               # вклад гибридного ранга в итоговый score
    rerank_query_tnved: bool = False  # добавлять ли ТН ВЭД-расширение в запрос реранкера
    skip_rerank: bool = False
    fp16: bool = False
    dump_debug: bool = True


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _timer(name: str, t0: float) -> float:
    t = time.perf_counter()
    log.info("%-28s %6.1f с", name, t - t0)
    return t


def tnved_expansions(cfg: Config, encoder: DenseEncoder, q_norm: list[str], q_lem: list[list[str]]) -> list[str]:
    """Для каждой декларации — строка из названий ближайших позиций ТН ВЭД."""
    raw = cfg.root / "tnved_knowledge.txt"
    cache = cfg.root / "data" / "tnved_index.jsonl"
    if not raw.exists() and not cache.exists():
        log.warning("tnved_knowledge.txt не найден — ТН ВЭД-обогащение отключено")
        return [""] * len(q_norm)
    entries = tnved.load_or_build_index(raw, cache)
    texts = [tnved.entry_text(e) for e in entries]
    t0 = time.perf_counter()
    bm25 = BM25([preprocess.lemmas(t) for t in texts])
    t0 = _timer("ТН ВЭД: BM25-индекс", t0)
    emb_path = cfg.root / "data" / "tnved_emb.npy"
    if emb_path.exists():
        emb = np.load(emb_path).astype(np.float32)
        assert emb.shape[0] == len(texts), "кэш эмбеддингов ТН ВЭД не совпадает с индексом; удалите data/tnved_emb.npy"
    else:
        log.info("ТН ВЭД: эмбеддинги %d позиций (кэша нет; запустите prepare.py, чтобы не считать при каждом запуске)", len(texts))
        emb = encoder.encode(texts, batch_size=64, show_progress=True)
        np.save(emb_path, emb.astype(np.float16))
    t0 = _timer("ТН ВЭД: эмбеддинги", t0)
    q_emb = encoder.encode(q_norm)
    dense_scores = q_emb @ emb.T
    out = []
    for i in range(len(q_norm)):
        fused = rrf([(bm25.scores(q_lem[i]), 1.0), (dense_scores[i], 1.0)], k=cfg.rrf_k)
        idx = topk(fused, cfg.tnved_top)
        names = []
        for j in idx:
            n = entries[j]["name"]
            if n not in names:
                names.append(n)
        out.append("; ".join(names))
    _timer("ТН ВЭД: расширение запросов", t0)
    return out


def run(cfg: Config) -> Path:
    t_start = t0 = time.perf_counter()
    decls = load_declarations(cfg.root)
    regs = load_regulations(cfg.root)
    docs = corpus.build_docs(regs)
    log.info("деклараций: %d, позиций НПА: %d, ссылок разрешено: %d",
             len(decls), len(docs), sum(len(d.ref_texts) for d in docs))

    q_norm = [preprocess.normalize(d["G31_1"]) for d in decls]
    q_lem = [preprocess.lemmas(d["G31_1"]) for d in decls]
    doc_bm25 = [preprocess.lemmas(d.text_for_bm25()) for d in docs]
    doc_model = [d.text_for_model() for d in docs]
    t0 = _timer("препроцессинг", t0)

    bm25 = BM25(doc_bm25)
    encoder = DenseEncoder(str(cfg.models_dir / "user-bge-m3"), cfg.device, fp16=cfg.fp16)
    t0 = _timer("загрузка энкодера", t0)

    expansions = tnved_expansions(cfg, encoder, q_norm, q_lem) if cfg.use_tnved else [""] * len(decls)
    t0 = time.perf_counter()

    doc_emb = encoder.encode(doc_model, show_progress=False)
    q_emb = encoder.encode(q_norm)
    exp_emb = encoder.encode(expansions) if cfg.use_tnved else None
    encoder.close()
    t0 = _timer("dense-поиск по НПА", t0)

    n_docs = len(docs)
    per_method: dict[str, list[np.ndarray]] = {"bm25": [], "dense": [], "hybrid": []}
    candidates: list[np.ndarray] = []
    hybrid_scores: list[np.ndarray] = []
    for i in range(len(decls)):
        s_bm25 = bm25.scores(q_lem[i])
        s_dense = q_emb[i] @ doc_emb.T
        lists = [(s_bm25, cfg.w_bm25), (s_dense, cfg.w_dense)]
        if cfg.use_tnved and expansions[i]:
            lists.append((bm25.scores(preprocess.lemmas(expansions[i])), cfg.w_tnved))
            lists.append((exp_emb[i] @ doc_emb.T, cfg.w_tnved))
        fused = rrf(lists, k=cfg.rrf_k)
        per_method["bm25"].append(topk(s_bm25, 50))
        per_method["dense"].append(topk(s_dense, 50))
        per_method["hybrid"].append(topk(fused, 50))
        candidates.append(topk(fused, max(cfg.rerank_k, TOP_N)))
        hybrid_scores.append(fused)
    t0 = _timer("BM25 + RRF", t0)

    if cfg.skip_rerank:
        logits = [np.full(len(c), np.nan, dtype=np.float32) for c in candidates]
    else:
        reranker = Reranker(str(cfg.models_dir / "bge-reranker-v2-m3"), cfg.device, fp16=cfg.fp16)
        t0 = _timer("загрузка реранкера", t0)
        rq = [q + (" " + e if cfg.rerank_query_tnved and e else "") for q, e in zip(q_norm, expansions)]
        elapsed = time.perf_counter() - t_start
        logits = rerank_all(reranker, rq, candidates, doc_model, max(60.0, cfg.time_budget_s - elapsed))
        t0 = _timer("реранкинг", t0)

    rows = []
    final_rank: list[np.ndarray] = []
    for i, d in enumerate(decls):
        cand = candidates[i]
        h = hybrid_scores[i][cand]
        h_norm = (h - h.min()) / (h.max() - h.min() + 1e-9)
        # Переранжированные кандидаты: sigmoid(logit) + λ·гибрид; остальные — ниже любого
        # переранжированного, в гибридном порядке (score в (-1, 0)).
        reranked = ~np.isnan(logits[i])
        score = np.where(reranked, _sigmoid(np.nan_to_num(logits[i])) + cfg.lam * h_norm, h_norm - 1.0)
        order = np.argsort(-score, kind="stable")
        chosen = cand[order][:TOP_N]
        chosen_scores = score[order][:TOP_N]
        final_rank.append(cand[order])
        for r, (j, s) in enumerate(zip(chosen, chosen_scores), start=1):
            rows.append({"declaration_id": d["declaration_id"], "rank": r,
                         "regulation_id": docs[j].regulation_id, "score": float(s)})

    path = write_predictions(rows, cfg.out_dir, [d["declaration_id"] for d in decls],
                             [d.regulation_id for d in docs])
    if cfg.dump_debug:
        dbg = cfg.out_dir / "debug_rankings.jsonl"
        with open(dbg, "w", encoding="utf-8") as f:
            for i, d in enumerate(decls):
                rec = {"declaration_id": d["declaration_id"], "tnved": expansions[i],
                       **{m: [docs[j].regulation_id for j in per_method[m][i]] for m in per_method},
                       "final": [docs[j].regulation_id for j in final_rank[i]]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("итого %.1f с; результат: %s", time.perf_counter() - t_start, path)
    return path
