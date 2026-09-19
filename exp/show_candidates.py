"""Помощник для ручной разметки.

python exp/show_candidates.py 70            # декларация №70: текст, текущая разметка, top-30 гибрида
python exp/show_candidates.py 70 --grep эвм # + все позиции НПА, содержащие слово
"""
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_utils import load_declarations, load_regulations

ap = argparse.ArgumentParser()
ap.add_argument("idx", type=int, help="номер строки в declarations.json (с нуля)")
ap.add_argument("--grep", help="поиск по тексту всех позиций (регистр не важен)")
ap.add_argument("--top", type=int, default=30)
a = ap.parse_args()

decs = load_declarations(ROOT); regs = {r["regulation_id"]: r for r in load_regulations(ROOT)}
d = decs[a.idx]
labels = {json.loads(l)["idx"]: json.loads(l) for l in open(ROOT / "data/dev_labels.jsonl", encoding="utf-8")}
dbg = {json.loads(l)["declaration_id"]: json.loads(l) for l in open(ROOT / "out/debug_rankings.jsonl", encoding="utf-8")}

print(f"=== [{a.idx}] {d['declaration_id']}\n{d['G31_1']}\n")
rel = labels.get(a.idx, {}).get("relevant", {})
print("текущая разметка:", rel or "нет", "\n")
print(f"top-{a.top} гибрида (★ — уже в разметке):")
for n, rid in enumerate(dbg[d["declaration_id"]]["hybrid"][:a.top], 1):
    mark = f"★{rel[rid]}" if rid in rel else "  "
    print(f" {n:2} {mark} {rid} ({regs[rid]['decree_number']}): {regs[rid]['npa'][:160]}")
if a.grep:
    print(f"\nпозиции со словом «{a.grep}»:")
    for rid, r in regs.items():
        if re.search(a.grep, r["npa"], re.I):
            print(f"    {rid} ({r['decree_number']}): {r['npa'][:160]}")
