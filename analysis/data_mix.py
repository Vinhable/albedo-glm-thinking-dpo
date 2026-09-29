import json, re, sys
from collections import Counter
sys.path.insert(0, r"C:\Users\VINH\Desktop\albedo\scripts")
from vinhable_dpo_data import supervised
EDIT = re.compile(r"(sed -i|cat\s*>|cat <<|>\s*[\w/.-]+\.(py|rs|go|js|ts|java|c|cpp|h)\b|python3? - <<|python3? -c .*write|apply_patch|patch |tee |perl -pi|str_replace|git apply)")
TEST = re.compile(r"(pytest|cargo test|go test|npm test|python3? -m unittest|make test|cargo build|cargo check|tox|mvn|gradle|python3? [\w/]*test)")
def kind(content):
    a = content.split("</think>")[-1]
    m = re.search(r"```bash\n(.*?)```", a, re.S)
    if not m: return "no_cmd"
    x = m.group(1)
    if "SUBMIT_TASK" in x or "git diff --cached" in x: return "submit"
    if EDIT.search(x): return "edit"
    if TEST.search(x): return "test/build"
    return "read/explore"
c = {"chosen": Counter(), "rejected": Counter()}; rows = 0; keys=None
for line in open(r"E:\albedo-storage-temp\vinhable-glm-pairs-20260927\single\train.jsonl", encoding="utf-8"):
    r = json.loads(line); rows += 1; keys = keys or list(r)
    for side in c:
        for m in r[side]:
            if m["role"] == "assistant" and supervised(m): c[side][kind(m["content"])] += 1
print("rows", rows, keys)
for side, k in c.items():
    t = sum(k.values()); print(side, t, " ".join(f"{n}={k[n]/t:.0%}" for n in ("read/explore","edit","test/build","submit","no_cmd")))
print()
from collections import defaultdict
by = defaultdict(lambda: defaultdict(Counter)); pos = defaultdict(list); full=Counter()
for line in open(r"E:\albedo-storage-temp\vinhable-glm-pairs-20260927\single\train.jsonl", encoding="utf-8"):
    r = json.loads(line); full[(r["chosen_side"], r["is_full_trajectory"])] += 1
    for side in ("chosen","rejected"):
        who = r[f"{side}_side"]
        a = [m for m in r[side] if m["role"]=="assistant"]
        for i,m in enumerate(a):
            if supervised(m):
                by[who][r["sample_phase"]][kind(m["content"])] += 1; pos[who].append(i+1)
print("chosen_side x full:", dict(full))
for who, d in by.items():
    for ph, k in d.items():
        t=sum(k.values()); print(who, ph, t, " ".join(f"{n}={k[n]/t:.0%}" for n in ("read/explore","edit","test/build","submit")))
import statistics as st
for who,p in pos.items(): print(who, "supervised turn index median", st.median(p), "p90", sorted(p)[int(.9*len(p))])
