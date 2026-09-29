import json, re, statistics as st, random
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
EDIT = re.compile(r"(sed -i|cat\s*>|cat <<|>\s*[\w/.-]+\.(py|rs|go|js|ts|java|c|cpp|h)\b|python3? - <<|python3? -c .*write|apply_patch|patch |tee |perl -pi|str_replace|git apply)")
def subcmds(x):
    x = re.sub(r"<<\s*'?(\w+)'?.*?\n\1", "HEREDOC", x, flags=re.S)
    return len([p for p in re.split(r"&&|;|\n|\|\|", x) if p.strip() and not p.strip().startswith("#")])
def feats(contents):
    sub, fe, n = [], None, 0
    for i, c in enumerate(contents):
        a = c.split("</think>")[-1]; m = re.search(r"```bash\n(.*?)```", a, re.S)
        if not m: continue
        x = m.group(1); n += 1
        if EDIT.search(x):
            fe = fe or i + 1
        elif "SUBMIT_TASK" not in x: sub.append(subcmds(x))
    return (st.mean(sub) if sub else None), fe
def within(groups, name):
    # per task: correlation sign between breadth and score across runs; pooled within-task deviations
    dx, dy = [], []
    for runs in groups.values():
        runs = [r for r in runs if r[0] is not None]
        if len(runs) < 2: continue
        mx = st.mean(r[0] for r in runs); my = st.mean(r[1] for r in runs)
        for b, s, _ in runs: dx.append(b - mx); dy.append(s - my)
    cov = sum(a*b for a, b in zip(dx, dy)); r = cov / (sum(a*a for a in dx) * sum(b*b for b in dy)) ** .5
    # score change per +1 subcmd/turn (within-task slope)
    slope = cov / sum(a*a for a in dx)
    print(f"{name:28s} runs {len(dx):5d} | within-task corr(breadth, score) {r:+.3f} | slope {slope:+.3f} score per +1 subcmd/turn")
# GLM reference runs: all tasks with >=2 runs (sample to keep it fast)
root = Path(r"E:\albedo-storage-temp\hf-albedo-reference-20260927\data\glm_5_2")
g = defaultdict(list); gfe = defaultdict(list)
for p in sorted(root.glob("*.parquet")):
    for r in pq.read_table(p, columns=["sample_id", "eval_run_id", "score", "messages"]).to_pylist():
        if r["score"] is None: continue
        msgs = r["messages"] if isinstance(r["messages"], list) else json.loads(r["messages"])
        b, fe = feats([m["content"] for m in msgs if m["role"] == "assistant"])
        g[(r["eval_run_id"], r["sample_id"])].append((b, float(r["score"]), fe))
within(g, "GLM reference (all tasks)")
# King in duel (2 rollouts per task)
D = Path(r"C:\Users\VINH\Desktop\able_e6_400_rollouts\sft-dpo-step70-vs-cxxvii-20260929")
for lab, stem in (("King duel", "king/cxxvii-king"), ("DPO duel", "challenger/dpo-step70"), ("SFT duel", "challenger/sft-step70")):
    k = defaultdict(list)
    for r in ("r1", "r2"):
        for l in open(D / f"{stem}-{r}.jsonl", encoding="utf-8"):
            d = json.loads(l)
            if d["trajectory_score"] is None: continue
            t = d.get("candidate_turns") or d.get("king_turns"); a = [x for x in t if x["role"] == "assistant"]
            gen = [x for x in a if x.get("score_target")] or a[-d["assistant_turns"]:]
            b, fe = feats([x["content"] for x in gen]); k[d["sample_id"]].append((b, d["trajectory_score"], fe))
    within(k, lab)
# between-model at task level already known; also: GLM runs bucketed by breadth
allr = [r for v in g.values() for r in v if r[0] is not None]
for lo, hi in ((0, 2), (2, 3), (3, 5), (5, 99)):
    s = [r for r in allr if lo <= r[0] < hi]
    fe = [r[2] for r in s if r[2]]
    print(f"GLM runs breadth [{lo},{hi}): n={len(s):5d} score {st.mean(r[1] for r in s):.3f} first_edit_med {st.median(fe) if fe else '-'}")
