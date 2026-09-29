import json, statistics as st
from pathlib import Path
D = Path(r"C:\Users\VINH\Desktop\able_e6_400_rollouts\sft-dpo-step70-vs-cxxvii-20260929")
man = [json.loads(l) for l in open(D / "duel-manifest.jsonl", encoding="utf-8")]
best, mean_, spread = [], [], []
for m in man:
    s = m["evaluation"]["trajectory_scores"]; k = [s.get("king_r1"), s.get("king_r2")]
    if None in k: continue
    best.append(max(k)); mean_.append(st.mean(k)); spread.append(abs(k[0] - k[1]))
print(f"King mean {st.mean(mean_):.3f} | King best-of-2 {st.mean(best):.3f} | mean |r1-r2| {st.mean(spread):.3f} | tasks with |r1-r2|>=0.2: {sum(x>=0.2 for x in spread)}/{len(spread)}")
# judge noise: same King trajectory, production score vs local rejudge
jd = []
for r in ("r1", "r2"):
    for l in open(D / f"king/cxxvii-king-{r}.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["trajectory_score"] is not None and d.get("production_score") is not None:
            jd.append(abs(d["trajectory_score"] - d["production_score"]))
print(f"judge re-read noise, same trajectory: mean |local - production| {st.mean(jd):.3f}, median {st.median(jd):.3f}, >=0.1: {sum(x>=0.1 for x in jd)}/{len(jd)}")
