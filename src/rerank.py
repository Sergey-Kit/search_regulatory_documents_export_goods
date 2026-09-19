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
    затем 11–20 и т.д. При исчерпании бюджета времени оставшиеся раунды пропускаются —
    качество деградирует равномерно по всем декларациям, а не «первые N деклараций целиком».
    """
    t0 = time.perf_counter()
    result = [np.full(len(c), np.nan, dtype=np.float32) for c in candidates]
    depth = max(len(c) for c in candidates)
    n_pairs = 0
    for start in range(0, depth, round_size):
        elapsed = time.perf_counter() - t0
        if start > 0 and elapsed > time_budget_s:
            log.warning("бюджет времени реранкера исчерпан (%.0f с): переранжированы позиции 1–%d из %d",
                        elapsed, start, depth)
            break
        for i, (q, cand) in enumerate(zip(queries, candidates)):
            sl = cand[start:start + round_size]
            if len(sl) == 0:
                continue
            result[i][start:start + len(sl)] = reranker.score(q, [doc_texts[j] for j in sl])
            n_pairs += len(sl)
        log.info("реранкер: глубина %d, %d пар, %.0f с", min(start + round_size, depth), n_pairs,
                 time.perf_counter() - t0)
    return result
