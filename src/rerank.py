"""Кросс-энкодер bge-reranker-v2-m3 с бюджетом времени."""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_dir: str, device: str, max_length: int = 512, batch_size: int = 8,
                 fp16: bool = False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval().to(device)
        # fp16 включается только явно: на GPU без тензорных ядер (GTX 16xx) он в разы медленнее fp32.
        if fp16 and device.startswith("cuda"):
            self.model.half()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.torch = torch

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(docs), self.batch_size):
                batch = docs[i:i + self.batch_size]
                enc = self.tok([query] * len(batch), batch, padding=True, truncation="longest_first",
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.view(-1).float().cpu().numpy()
                out.append(logits)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def rerank_all(reranker: Reranker, queries: list[str], candidates: list[np.ndarray],
               doc_texts: list[str], time_budget_s: float, round_size: int = 10) -> list[np.ndarray]:
    """Логиты реранкера по кандидатам каждой декларации; NaN — кандидат не переранжирован.

    Кандидаты обрабатываются раундами по глубине: сначала позиции 1–10 у всех деклараций,
    затем 11–20 и т.д. Если бюджет времени исчерпан посреди раунда, раунд откатывается, и
    остаются только полностью завершённые раунды — качество деградирует равномерно по всем
    декларациям, а не «первые N деклараций целиком». Результат детерминирован при любой
    завершённой глубине.
    """
    t0 = time.perf_counter()
    result = [np.full(len(c), np.nan, dtype=np.float32) for c in candidates]
    depth = max(len(c) for c in candidates)
    n_pairs = 0
    probe = 5  # после стольких деклараций раунда экстраполируем его длительность
    for start in range(0, depth, round_size):
        round_res = []
        t_round = time.perf_counter()
        budget_left = time_budget_s - (t_round - t0)
        for n_done, (q, cand) in enumerate(zip(queries, candidates)):
            elapsed = time.perf_counter() - t0
            projected = (time.perf_counter() - t_round) / n_done * len(queries) if n_done >= probe else 0.0
            if elapsed > time_budget_s or projected > budget_left:
                log.warning("бюджет времени реранкера (%.0f с) не позволяет завершить раунд %d–%d "
                            "(прогноз %.0f с, прошло %.0f с): раунд откачен, переранжированы позиции 1–%d из %d",
                            time_budget_s, start + 1, start + round_size, projected, elapsed, start, depth)
                return result
            sl = cand[start:start + round_size]
            round_res.append(reranker.score(q, [doc_texts[j] for j in sl]) if len(sl) else None)
            n_pairs += len(sl)
        for i, r in enumerate(round_res):
            if r is not None:
                result[i][start:start + len(r)] = r
        log.info("реранкер: глубина %d, %d пар, %.0f с", min(start + round_size, depth), n_pairs,
                 time.perf_counter() - t0)
    return result
