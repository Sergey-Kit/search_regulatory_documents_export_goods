"""Однократная подготовка окружения (разрешена сетью): модели → ./models, кэш ТН ВЭД → ./data.

python prepare.py            # скачать модели (или скопировать из HF-кэша), построить индекс ТН ВЭД
python prepare.py --tnved    # дополнительно предрассчитать эмбеддинги ТН ВЭД (нужны только для run.py --tnved)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    "user-bge-m3": ("deepvk/USER-bge-m3", "0cc6cfe48e260fb0474c753087a69369e88709ae"),
    "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "ru-en-rosberta": ("ai-forever/ru-en-RoSBERTa", "89fb1651989adbb1cfcfdedafd7d102951ad0555"),
}
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.json", "merges.txt"]


def download_models(models_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    manifest = {}
    for local, (repo, rev) in MODELS.items():
        dst = models_dir / local
        if (dst / "config.json").exists() and (dst / "model.safetensors").exists():
            print(f"[skip] {local}: уже есть")
        else:
            print(f"[download] {repo}@{rev[:8]} → {dst}")
            snapshot_download(repo, revision=rev, local_dir=str(dst), allow_patterns=FILES)
        manifest[local] = {"repo": repo, "revision": rev}
    (models_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_tnved_cache(with_emb: bool) -> None:
    import numpy as np

    from src import tnved
    from src.retrieval import DenseEncoder

    raw = ROOT / "tnved_knowledge.txt"
    if not raw.exists():
        print("[skip] tnved_knowledge.txt не найден")
        return
    entries = tnved.load_or_build_index(raw, ROOT / "data" / "tnved_index.jsonl")
    print(f"[ok] индекс ТН ВЭД: {len(entries)} позиций")
    if not with_emb:
        return
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = DenseEncoder(str(ROOT / "models" / "user-bge-m3"), device)
    emb = enc.encode([tnved.entry_text(e) for e in entries], batch_size=64, show_progress=True)
    np.save(ROOT / "data" / "tnved_emb.npy", emb.astype(np.float16))
    print(f"[ok] эмбеддинги ТН ВЭД: {emb.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tnved", action="store_true", help="предрассчитать эмбеддинги ТН ВЭД (для run.py --tnved)")
    a = p.parse_args()
    download_models(ROOT / "models")
    build_tnved_cache(with_emb=a.tnved)
    sys.exit(0)
