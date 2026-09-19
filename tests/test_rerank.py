import numpy as np

from src.rerank import rerank_all


class SlowFakeReranker:
    """Возвращает -index как логит; каждая оценка «стоит» dt секунд (через подмену времени)."""

    def __init__(self, clock):
        self.clock = clock

    def score(self, query, docs):
        self.clock[0] += 1.0
        return np.array([float(len(d)) for d in docs], dtype=np.float32)


def test_rounds_and_rollback(monkeypatch):
    clock = [0.0]
    import src.rerank as rr
    monkeypatch.setattr(rr.time, "perf_counter", lambda: clock[0])
    docs = [str(i) * (i + 1) for i in range(30)]
    cands = [np.arange(30), np.arange(30)]
    # бюджет 2.5 «секунды»: раунд 1 (2 вызова) успевает, раунд 2 прерывается и откатывается
    res = rerank_all(SlowFakeReranker(clock), ["q", "q"], cands, docs, time_budget_s=2.5, round_size=10)
    for r in res:
        assert not np.isnan(r[:10]).any() and np.isnan(r[10:]).all()
    clock[0] = 0.0
    res = rerank_all(SlowFakeReranker(clock), ["q", "q"], cands, docs, time_budget_s=1e9, round_size=10)
    assert all(not np.isnan(r).any() for r in res)
