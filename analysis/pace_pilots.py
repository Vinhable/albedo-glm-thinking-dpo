import json, re, statistics as st
src = open("pace.py", encoding="utf-8").read()
exec(src.split("for lab in")[0].split('exec(open("duel_diag.py"')[0] + src.split('exec(open("duel_diag.py", encoding="utf-8").read().split(\'print("== score distribution\')[0])')[1].split("for lab in")[0])
for v in ["", "-v2", "-v3", "-v4", "-v5"]:
    p = rf"E:\albedo-storage-temp\glm-thinking-pilot{v}-20260927\results.jsonl"
    L = []; sc = []
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        t = [x for x in d.get("turns", []) if x.get("role") == "assistant" and "action" in x]
        if not t: continue
        L.append([(x.get("reasoning") or "") + "\n</think>\n\n" + (x.get("action") or x.get("content") or "") for x in t])
        if d.get("glm_score") is not None: sc.append(d["glm_score"])
    print(f"pilot{v or '-v1'} n={len(L)} glm_score={st.mean(sc) if sc else float('nan'):.3f}", end=" ")
    stats(L, "")
