import json, re, statistics as st, random
from collections import Counter, defaultdict
exec(open("duel_diag.py", encoding="utf-8").read().split('print("== score distribution')[0])
exec(open("glm_edit_submit.py", encoding="utf-8").read().split("# ---------- 1.")[0].split("sys.path.insert")[0])
import re as _re
EDIT2 = _re.compile(r"(sed -i|cat\s*>|cat <<|>\s*[\w/.-]+\.(py|rs|go|js|ts|java|c|cpp|h)\b|python3? - <<|python3? -c .*write|apply_patch|patch |tee |perl -pi|str_replace|git apply)")
tasks = []
for sid, v in rows.items():
    qs = v["king"][0]["questions"]; tg = Counter(q["tag"] for q in qs); n = len(qs)
    tasks.append((sid, tg, n, v))
print("tasks", len(tasks))
print("tasks with 0 action questions:", sum(t[1]["reference:action"] == 0 for t in tasks),
      "| 0 verification:", sum(t[1]["reference:verification"] == 0 for t in tasks),
      "| only explore(+claims):", sum(t[1]["reference:action"] == 0 and t[1]["reference:verification"] == 0 for t in tasks))
shares = defaultdict(list)
for sid, tg, n, v in tasks:
    for k in ("reference:explore", "reference:action", "reference:verification", "reference:claims"): shares[k].append(tg[k] / n)
print("weight share per task (mean, min, max):", {k.split(':')[1]: (round(st.mean(s), 2), round(min(s), 2), round(max(s), 2)) for k, s in shares.items()})
# action questions: how many mention changing code
act = [q for t in tasks for q in t[3]["king"][0]["questions"] if q["tag"] == "reference:action"]
chg = [q for q in act if _re.search(r"\b(change|modif|add|remov|replac|implement|updat|fix|rewrit|creat|edit)", q["text"], _re.I)]
print(f"action questions {len(act)}, phrased as a code change {len(chg)} ({len(chg)/len(act):.0%})")
print("claims examples:"); [print("  -", q["text"][:170]) for q in [q for t in tasks[:6] for q in t[3]["king"][0]["questions"] if q["tag"] == "reference:claims"][:5]]
print("verification examples:"); [print("  -", q["text"][:170]) for q in [q for t in tasks[:6] for q in t[3]["king"][0]["questions"] if q["tag"] == "reference:verification"][:3]]
# empirical: ceiling without edit / without submit (all 600 trajectories)
def gt(d):
    turns = d.get("candidate_turns") or d.get("king_turns") or []
    a = [t for t in turns if t.get("role") == "assistant"]; g = [t for t in a if t.get("score_target")]
    return g if g else a[-d["assistant_turns"]:]
def cmds(d):
    out = []
    for t in gt(d):
        m = _re.search(r"```bash\n(.*?)```", t["content"].split("</think>")[-1], _re.S); out.append(m.group(1) if m else "")
    return out
B = defaultdict(list)
for sid, tg, n, v in tasks:
    for lab in ("king", "dpo", "sft"):
        for d in v[lab]:
            if d["trajectory_score"] is None: continue
            c = cmds(d); e = any(EDIT2.search(x) for x in c); s = any(d["submit_marker"] in x for x in c)
            B[(e, s)].append(d["trajectory_score"])
for (e, s), xs in sorted(B.items()):
    print(f"edit={e!s:5} submit={s!s:5} n={len(xs):3d} mean={st.mean(xs):.3f} >=0.9: {sum(x>=0.9 for x in xs)} >=0.97: {sum(x>=0.97 for x in xs)} max={max(xs):.3f}")
