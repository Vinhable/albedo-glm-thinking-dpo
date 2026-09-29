"""How much GLM data shows edit / submit behaviour: full trajectories (batch 300) and dataset rows."""
import json, re, statistics as st, sys
from collections import Counter, defaultdict

sys.path.insert(0, r"C:\Users\VINH\Desktop\albedo\scripts")
from vinhable_dpo_data import supervised  # noqa: E402

EDIT = re.compile(r"(sed -i|cat\s*>|cat <<|>\s*[\w/.-]+\.(py|rs|go|js|ts|java|c|cpp|h)\b|python3? - <<|python3? -c .*write|"
                  r"apply_patch|patch |tee |perl -pi|str_replace|git apply)")
TEST = re.compile(r"(pytest|cargo test|go test|npm test|python3? -m unittest|make test|cargo build|cargo check|tox|mvn|gradle|"
                  r"python3? [\w/]*test)")


def kind(action):
    m = re.search(r"```bash\n(.*?)```", action or "", re.S)
    if not m:
        return "no_cmd"
    x = m.group(1)
    if "SUBMIT_TASK" in x or "git diff --cached" in x:
        return "submit"
    if EDIT.search(x):
        return "edit"
    if TEST.search(x):
        return "test"
    return "explore"


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


# ---------- 1. full GLM trajectories, batch 300
print("== GLM full trajectories (batch 300, generated turns only)")
trajs = []
for line in open(r"E:\albedo-storage-temp\glm-thinking-batch300-20260927\merged.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if d.get("status") != "ok" or d.get("glm_score") is None:
        continue
    gen = [t for t in d["turns"] if t.get("role") == "assistant" and "action" in t]
    kinds = [kind(t.get("action") or t.get("content")) for t in gen]
    trajs.append({"phase": d["sample_phase"], "source": d["source"], "stop": d.get("stop"), "horizon": d["horizon"],
                  "kinds": kinds, "glm": d["glm_score"], "king": d["king_mean"],
                  "edit": "edit" in kinds, "submit": "submit" in kinds,
                  "first_edit": kinds.index("edit") + 1 if "edit" in kinds else None})
n = len(trajs)
print("trajectories", n, "stop:", Counter(t["stop"] for t in trajs))
allk = Counter(k for t in trajs for k in t["kinds"]); tot = sum(allk.values())
print("action mix:", " ".join(f"{k}={allk[k] / tot:.0%}" for k in ("explore", "edit", "test", "submit", "no_cmd")))
print("with edit:", pct(sum(t["edit"] for t in trajs), n), " with submit:", pct(sum(t["submit"] for t in trajs), n),
      " edit AND submit:", pct(sum(t["edit"] and t["submit"] for t in trajs), n))
fe = [t["first_edit"] for t in trajs if t["first_edit"]]
print("first edit turn median", st.median(fe), "p75", sorted(fe)[int(.75 * len(fe))])
for ph in ("cold", "pre_edit", "at_edit"):
    s = [t for t in trajs if t["phase"] == ph]
    if s:
        print(f"  {ph:9s} n={len(s):3d} edit {pct(sum(t['edit'] for t in s), len(s))} submit {pct(sum(t['submit'] for t in s), len(s))}"
              f" horizon12 {sum(t['horizon'] == 12 for t in s)}")

print("\n-- score by behaviour (GLM score / King mean / share with GLM-King >= 0.15)")
buckets = {"edit+submit": lambda t: t["edit"] and t["submit"], "edit, no submit": lambda t: t["edit"] and not t["submit"],
           "no edit": lambda t: not t["edit"]}
for name, f in buckets.items():
    s = [t for t in trajs if f(t)]
    if s:
        win = sum(t["glm"] - t["king"] >= 0.15 for t in s)
        print(f"  {name:16s} n={len(s):3d} glm={st.mean(t['glm'] for t in s):.3f} king={st.mean(t['king'] for t in s):.3f} "
              f"glm_win>=0.15 {pct(win, len(s))}")

print("\n-- the pool we train on: GLM wins (glm - king >= 0.15)")
w = [t for t in trajs if t["glm"] - t["king"] >= 0.15]
print("  n", len(w), "edit", pct(sum(t["edit"] for t in w), len(w)), "submit", pct(sum(t["submit"] for t in w), len(w)),
      "edit+submit", pct(sum(t["edit"] and t["submit"] for t in w), len(w)))

# ---------- 2. dataset rows
for ds in ("single", "double"):
    print(f"\n== vinhable_{ds} rows (supervised turns of the GLM side)")
    rows = []
    for split in ("train", "dev"):
        for line in open(rf"E:\albedo-storage-temp\vinhable-glm-pairs-20260927\{ds}\{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            glm_side = "chosen" if r["chosen_side"].startswith("glm") else "rejected"
            ks = [kind(m["content"].split("</think>")[-1]) for m in r[glm_side] if m["role"] == "assistant" and supervised(m)]
            rows.append({"dir": r["direction"], "pair": r["pair_id"], "ks": ks, "phase": r["sample_phase"]})
    for d in ("glm_over_king", "king_over_glm"):
        s = [r for r in rows if r["dir"] == d]
        pairs = defaultdict(set)
        for r in s:
            pairs[r["pair"]].update(r["ks"])
        e = sum("edit" in r["ks"] for r in s); sb = sum("submit" in r["ks"] for r in s)
        es = sum(("edit" in r["ks"]) or ("submit" in r["ks"]) or ("test" in r["ks"]) for r in s)
        print(f"  {d}: rows {len(s)} | group has edit {pct(e, len(s))} | submit {pct(sb, len(s))} | edit/test/submit {pct(es, len(s))}"
              f" | pairs {len(pairs)}, pairs with any edit {sum('edit' in v for v in pairs.values())}, any submit {sum('submit' in v for v in pairs.values())}")
