import sys, json; from pathlib import Path; import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_declarations; from src import preprocess; from src.retrieval import DenseEncoder
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "exp/cache"
names = [preprocess.product_name(d["G31_1"]) for d in load_declarations(ROOT)]
json.dump(names, open(C / "names.json", "w"), ensure_ascii=False)
enc = DenseEncoder(str(ROOT / "models/user-bge-m3"), "cuda")
np.save(C / "q_name.npy", enc.encode(names))
