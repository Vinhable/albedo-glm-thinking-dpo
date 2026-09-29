import json, re, statistics as st
from collections import defaultdict
exec(open("duel_diag.py", encoding="utf-8").read().split('print("== score distribution')[0])
EDIT = re.compile(r"(sed -i|cat\s*>|cat <<|>\s*[\w/.-]+\.(py|rs|go|js|ts|java|c|cpp|h)\b|python3? - <<|python3? -c .*write|apply_patch|patch |tee |perl -pi|str_replace|git apply)")
def subcmds(x):
    x = re.sub(r"<<\s*'?(\w+)'?.*?\n\1", "HEREDOC", x, flags=re.S)
    return len([p for p in re.split(r"&&|;|\n|\|\|", x) if p.strip() and not p.strip().startswith("#")])
def stats(turn_lists, label):
    th, sub, files, first_edit, turns_before_edit_think = [], [], [], [], []
    for turns in turn_lists:
        fe = None
        for i, c in enumerate(turns):
            h, _, a = c.rpartition("</think>") if "</think>" in c else ("", "", c)
            m = re.search(r"```bash\n(.*?)```", a, re.S)
            if not m: continue
            x = m.group(1)
            if EDIT.search(x) and fe is None: fe = i + 1
            if not EDIT.search(x) and "SUBMIT_TASK" not in x:
                th.append(len(h)); sub.append(subcmds(x))
                files.append(len(set(re.findall(r"[\w./-]+\.(?:py|rs|go|js|ts|java|c|h|toml|cfg|md)\b", x))))
        if fe: first_edit.append(fe)
    print(f"{label:22s} explore turns {len(sub):5d} | subcmds/turn mean {st.mean(sub):.2f} (1 only: {sum(s==1 for s in sub)/len(sub):.0%}) "
          f"| files touched/turn {st.mean(files):.2f} | think chars/explore turn {st.mean(th):.0f} | first edit med {st.median(first_edit) if first_edit else '-'}")
for lab in ("king", "dpo", "sft"):
    L = []
    for v in rows.values():
        for d in v[lab]:
            turns = d.get("candidate_turns") or d.get("king_turns")
            a = [t for t in turns if t["role"] == "assistant"]; g = [t for t in a if t.get("score_target")] or a[-d["assistant_turns"]:]
            L.append([t["content"] for t in g])
    stats(L, f"duel {lab}")
L = []
for line in open(r"E:\albedo-storage-temp\glm-thinking-batch300-20260927\merged.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if d.get("status") != "ok": continue
    L.append([(t.get("reasoning") or "") + "\n</think>\n\n" + (t.get("action") or t.get("content") or "") for t in d["turns"] if t.get("role") == "assistant" and "action" in t])
stats(L, "GLM batch300")
