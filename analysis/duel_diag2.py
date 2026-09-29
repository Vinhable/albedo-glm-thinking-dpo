import json, re, statistics as st
from collections import defaultdict, Counter
exec(open("duel_diag.py", encoding="utf-8").read().split('print("== score distribution')[0])
def cmd(a):
    m = re.search(r"```bash\n(.*?)```", a, re.S); return m.group(1).strip() if m else None
EDIT = re.compile(r"(sed -i|cat\s*>|cat <<|>\s*[\w/.-]+\.(py|rs|go|js|ts|java|c|cpp|h)\b|python3? - <<|python3? -c .*write|apply_patch|patch |tee |perl -pi|str_replace|git apply)")
TEST = re.compile(r"(pytest|cargo test|go test|npm test|python3? -m unittest|make test|cargo build|cargo check|tox|mvn|gradle|python3? [\w/]*test)")
for lab in FILES:
    c = Counter(); rep = 0; n = 0; edit_roll = 0; test_roll = 0; first_edit = []
    for v in rows.values():
        for d in v[lab]:
            g = gen_turns(d); cmds = [cmd(think_action(t["content"])[1]) for t in g]
            seen = set(); has_e = has_t = False
            for i, x in enumerate(cmds):
                n += 1
                if x is None: c["no_cmd"] += 1; continue
                if x in seen: rep += 1
                seen.add(x)
                if d["submit_marker"] in x: c["submit"] += 1
                elif EDIT.search(x): c["edit"] += 1; (first_edit.append(i+1) if not has_e else None); has_e = True
                elif TEST.search(x): c["test/build"] += 1; has_t = True
                else: c["read/explore"] += 1
            edit_roll += has_e; test_roll += has_t
    tot = sum(c.values())
    print(f"{lab:5s} " + " ".join(f"{k}={c[k]/tot:.0%}" for k in ("read/explore","edit","test/build","submit","no_cmd"))
          + f" | rollouts with edit={edit_roll}/200 with test={test_roll}/200 first_edit_turn_med={st.median(first_edit) if first_edit else '-'} repeated_cmd={rep/n:.1%}")
# phase mix of gen turns: does cold sft fail to reach edit?
print()
for ph in ("cold","pre_edit","at_edit"):
    out=[]
    for lab in FILES:
        e=s=k=0
        for v in rows.values():
            for d in v[lab]:
                if d["sample_phase"]!=ph: continue
                k+=1; g=[cmd(think_action(t["content"])[1]) or "" for t in gen_turns(d)]
                e+=any(EDIT.search(x) for x in g); s+=any(d["submit_marker"] in x for x in g)
        out.append(f"{lab} edit {e}/{k} submit {s}/{k}")
    print(ph, " | ".join(out))
