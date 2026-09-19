"""Sparse (BM25) и dense (bge-m3) поиск, слияние RRF."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    """Okapi BM25 на предварительно токенизированных документах."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n else 1.0
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.n - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        # инвертированный индекс: term -> [(doc_idx, tf)]
        self.index: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.index.setdefault(t, []).append((i, f))

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_len)
        for t, qf in Counter(query).items():
            postings = self.index.get(t)
            if not postings:
                continue
            idf = self.idf[t]
            for i, f in postings:
                out[i] += idf * f * (self.k1 + 1) / (f + norm[i])
        return out


class DenseEncoder:
    """Обёртка над USER-bge-m3: CLS-пулинг + L2-нормализация (как у bge-m3)."""

    def __init__(self, model_dir: str, device: str, max_seq_length: int = 512, fp16: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer, models

        tr = models.Transformer(model_dir, max_seq_length=max_seq_length)
        pool = models.Pooling(tr.get_word_embedding_dimension(), pooling_mode="cls")
        self.model = SentenceTransformer(modules=[tr, pool, models.Normalize()], device=device)
        if fp16 and device.startswith("cuda"):
            self.model.half()
        self.device = device
        self.torch = torch

    def encode(self, texts: list[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
        # Сортировка по длине внутри encode уменьшает паддинг; результат возвращается в исходном порядке.
        emb = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return emb.astype(np.float32)

    def close(self) -> None:
        import gc
        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


def rrf(rankings: list[tuple[np.ndarray, float]], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion. rankings — список (массив score по всем документам, вес)."""
    n = len(rankings[0][0])
    fused = np.zeros(n, dtype=np.float64)
    for scores, w in rankings:
        if w == 0:
            continue
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(1, n + 1)
        fused += w / (k + ranks)
    return fused


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]
