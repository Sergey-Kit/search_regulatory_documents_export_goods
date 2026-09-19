import sys; from pathlib import Path; import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_regulations; from src.corpus import build_docs; from src.retrieval import DenseEncoder
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "exp/cache"
docs = build_docs(load_regulations(ROOT))
texts = [d.text_with_category() for d in docs]
enc = DenseEncoder(str(ROOT / "models/user-bge-m3"), "cuda"); np.save(C / "doc_D.npy", enc.encode(texts)); enc.close()
enc = DenseEncoder(str(ROOT / "models/ru-en-rosberta"), "cuda"); np.save(C / "ros_doc_D.npy", enc.encode(["search_document: " + t for t in texts])); enc.close()
print("ok")
