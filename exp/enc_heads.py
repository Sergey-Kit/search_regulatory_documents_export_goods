import sys; from pathlib import Path; import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_declarations; from src import preprocess; from src.retrieval import DenseEncoder
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "exp/cache"
q = [preprocess.normalize(d["G31_1"]) for d in load_declarations(ROOT)]
enc = DenseEncoder(str(ROOT / "models/user-bge-m3"), "cuda")
for n in [150, 400]:
    np.save(C / f"q_head{n}.npy", enc.encode([x[:n] for x in q]))
