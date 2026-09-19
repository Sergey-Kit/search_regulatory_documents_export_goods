import sys; from pathlib import Path; import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_declarations, load_regulations; from src.corpus import build_docs; from src import preprocess
from src.retrieval import DenseEncoder
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "exp/cache"
decs = load_declarations(ROOT); docs = build_docs(load_regulations(ROOT))
enc = DenseEncoder(str(ROOT / "models/ru-en-rosberta"), "cuda")  # CLS + normalize — совпадает с 1_Pooling модели
q = [preprocess.normalize(d["G31_1"]) for d in decs]
np.save(C / "ros_doc_C.npy", enc.encode(["search_document: " + d.body for d in docs]))
np.save(C / "ros_doc_B.npy", enc.encode(["search_document: " + d.text_for_bm25() for d in docs]))
np.save(C / "ros_q_full.npy", enc.encode(["search_query: " + x for x in q]))
np.save(C / "ros_q_head.npy", enc.encode(["search_query: " + x[:250] for x in q]))
print("ok")
