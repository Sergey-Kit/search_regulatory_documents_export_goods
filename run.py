"""Точка входа: python run.py --out ./out

Сетевые запросы запрещены: до импорта transformers включаем оффлайн-режим HF, модели
берутся только из ./models (см. prepare.py).
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

from src.pipeline import Config, run

ROOT = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="Ранжирование НПА по тексту таможенной декларации")
    p.add_argument("--out", default="./out", help="каталог для predictions.csv")
    p.add_argument("--data", default=str(ROOT), help="каталог с declarations/regulations")
    p.add_argument("--models", default=str(ROOT / "models"), help="каталог с локальными моделями")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--rerank-k", type=int, default=20, help="глубина переранжирования (кандидатов из гибрида)")
    p.add_argument("--time-budget", type=float, default=1200.0, help="общий бюджет времени на реранкинг, с")
    p.add_argument("--tnved", action="store_true", help="включить ТН ВЭД-обогащение запроса (по умолчанию выключено)")
    p.add_argument("--no-rerank", action="store_true", help="только гибридный поиск (для абляций)")
    p.add_argument("--w-tnved", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--rerank-query-tnved", action="store_true")
    p.add_argument("--fp16", action="store_true", help="fp16 на GPU (быстрее только при наличии тензорных ядер)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA недоступна, используется CPU")
        device = "cpu"
    logging.info("устройство: %s, torch %s", device, torch.__version__)

    models_dir = Path(a.models)
    for name in ("user-bge-m3", "bge-reranker-v2-m3"):
        if not (models_dir / name / "config.json").exists():
            logging.error("модель %s не найдена в %s — выполните `python prepare.py`", name, models_dir)
            return 2

    cfg = Config(
        root=Path(a.data), out_dir=Path(a.out), models_dir=models_dir, device=device,
        rerank_k=a.rerank_k, time_budget_s=a.time_budget,
        use_tnved=a.tnved, w_tnved=a.w_tnved, lam=a.lam,
        rerank_query_tnved=a.rerank_query_tnved, skip_rerank=a.no_rerank, fp16=a.fp16,
    )
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
