import numpy as np

from src.retrieval import BM25, rrf, topk


def test_bm25_prefers_matching_doc():
    docs = [["насос", "струйный"], ["микросхема", "интегральный"], ["насос", "вакуумный", "насос"]]
    bm = BM25(docs)
    s = bm.scores(["насос", "струйный"])
    assert int(np.argmax(s)) == 0 and s[1] == 0


def test_rrf_and_topk():
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 2.0, 3.0])
    fused = rrf([(a, 1.0), (b, 1.0)])
    assert fused[0] == fused[2]
    assert abs(fused[1] - 2 / 62) < 1e-12
    assert list(topk(a, 2)) == [0, 1]
