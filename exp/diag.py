import sys; from pathlib import Path; import numpy as np
sys.argv=['x']; ROOT=Path(__file__).resolve().parents[1]
exec(open(ROOT/'exp/eval_variants.py').read().split('base = dict')[0])
i=int(sys.argv[1]) if len(sys.argv)>1 else 127
for rid_name in ["NPA0552","NPA0446"]:
    rid=ids.index(rid_name)
    def rank(scores): return int((np.argsort(-scores)==rid).nonzero()[0][0])+1
    print(rid_name, "bm25cat:", rank(bmD.scores(q_lem[i])), "| bge full/head:", rank(Q["full"][i]@D["D"].T), rank(Q["head"][i]@D["D"].T),
          "| ros full/head:", rank(ROS["q_full"][i]@ROS["doc_D"].T), rank(ROS["q_head"][i]@ROS["doc_D"].T), "| char full/head:", rank(CHAR[i]), rank(CHAR_H[i]))
