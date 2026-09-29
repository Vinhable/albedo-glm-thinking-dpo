import json, re, statistics as st
EDIT = re.compile(r"(sed -i|cat\s*>|cat <<|>\s*[\w/.-]+\.(py|rs|go|js|ts|java|c|cpp|h)\b|python3? - <<|python3? -c .*write|apply_patch|patch |tee |perl -pi|str_replace|git apply)")
TEST = re.compile(r"(pytest|cargo test|go test|npm test|python3? -m unittest|make test|cargo build|cargo check|tox|mvn|gradle|python3? [\w/]*(test|repro)\w*\.py)")
def subcmds(x):
    x = re.sub(r"<<\s*'?(\w+)'?.*?\n\1", "HEREDOC", x, flags=re.S)
    return len([p for p in re.split(r"&&|;|\n|\|\|", x) if p.strip() and not p.strip().startswith("#")])
def feats(side):
    a = [t for t in side if t["role"] == "assistant"]
    f = {"turns": len(a), "edit": 0, "test": 0, "submit": 0, "first_edit": None, "think": [], "sub": [], "repeat": 0}
    seen = set()
    for i, t in enumerate(a):
        c = t["content"]; h, _, act = c.partition("</think>")
        f["think"].append(len(h))
        m = re.search(r"```bash\n(.*?)```", act, re.S)
        if not m: continue
        x = m.group(1).strip()
        if x in seen: f["repeat"] += 1
        seen.add(x)
        if "SUBMIT" in x or "git diff --cached" in x: f["submit"] = 1
        elif EDIT.search(x): f["edit"] += 1; f["first_edit"] = f["first_edit"] or i + 1
        elif TEST.search(x): f["test"] += 1
        else: f["sub"].append(subcmds(x))
    return f
P = {"chosen": [], "rejected": []}
# trajectory-level: rebuild from full rows (is_full_trajectory rows hold whole sides)
for split in ("train", "dev"):
    for l in open(rf"E:\albedo-storage-temp\king-selfpairs-20260929\rows\{split}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r["is_full_trajectory"]: continue
        for s in P: P[s].append(feats(r[s]))
n = len(P["chosen"]); print("pairs", n)
def row(name, fn):
    c = [fn(f) for f in P["chosen"]]; r = [fn(f) for f in P["rejected"]]
    c = [x for x in c if x is not None]; r = [x for x in r if x is not None]
    print(f"{name:34s} chosen {st.mean(c):7.3f} | rejected {st.mean(r):7.3f}")
row("assistant turns", lambda f: f["turns"])
row("has edit", lambda f: float(f["edit"] > 0))
row("edit turns", lambda f: f["edit"])
row("first edit turn (if any)", lambda f: f["first_edit"])
row("has test/repro run", lambda f: float(f["test"] > 0))
row("test/repro turns", lambda f: f["test"])
row("submitted", lambda f: f["submit"])
row("subcmds per explore turn", lambda f: st.mean(f["sub"]) if f["sub"] else None)
row("think chars per turn", lambda f: st.mean(f["think"]) if f["think"] else None)
row("repeated identical command", lambda f: f["repeat"])
