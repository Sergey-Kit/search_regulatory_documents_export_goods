import sys, json; from pathlib import Path
sys.argv=['x']; ROOT=Path(__file__).resolve().parents[1]
exec(open(ROOT/'exp/eval_variants.py').read().split('base = dict')[0])
def ranking(i):
    s_b = bmD.scores(q_lem[i]); lists=[(s_b,1.0)]
    for qk in ["full","head"]:
        lists.append((Q[qk][i] @ D["D"].T,1.0)); lists.append((ROS["q_"+qk][i] @ ROS["doc_D"].T,1.0))
    lists += [(CHAR[i],1.0),(CHAR_H[i],1.0)]
    return [ids[j] for j in topk(rrf(lists,k=60),562)]
for i, rel in labels.items():
    r = ranking(i)
    for rid, g in rel.items():
        pos = r.index(rid)+1
        if g==2 and pos>10: print(i, rid, "позиция", pos, "|", decs[i]["G31_1"][:60])
