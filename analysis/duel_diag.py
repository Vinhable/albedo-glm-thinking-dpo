import json, re, statistics as st, random
from collections import defaultdict
from pathlib import Path

D = Path(r"C:\Users\VINH\Desktop\able_e6_400_rollouts\sft-dpo-step70-vs-cxxvii-20260929")
FILES = {"sft": "challenger/sft-step70", "dpo": "challenger/dpo-step70", "king": "king/cxxvii-king"}
rows = defaultdict(dict)  # sample -> label -> [r1, r2]
for lab, stem in FILES.items():
    for r in ("r1", "r2"):
        for line in open(D / f"{stem}-{r}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            rows[d["sample_id"]].setdefault(lab, []).append(d)

def gen_turns(d):
    turns = d.get("candidate_turns") or d.get("king_turns") or []
    a = [t for t in turns if t.get("role") == "assistant"]
    g = [t for t in a if t.get("score_target")]
    return g if g else a[-d["assistant_turns"]:]

def think_action(c):
    if "</think>" in c:
        h, _, a = c.partition("</think>")
        return h, a
    return "", c

def q(xs, p):
    xs = sorted(xs); return xs[int(p * (len(xs) - 1))] if xs else float("nan")

print("== score distribution per rollout")
for lab in FILES:
    s = [d["trajectory_score"] for v in rows.values() for d in v[lab] if d["trajectory_score"] is not None]
    print(f"{lab:5s} n={len(s)} mean={st.mean(s):.3f} med={st.median(s):.3f} p10={q(s,.1):.3f} p25={q(s,.25):.3f} "
          f"<0.5={sum(x<0.5 for x in s)} <0.3={sum(x<0.3 for x in s)} >=0.9={sum(x>=0.9 for x in s)}")

print("\n== replica noise: mean |r1-r2| per task")
for lab in FILES:
    g = [abs(v[lab][0]["trajectory_score"] - v[lab][1]["trajectory_score"]) for v in rows.values()
         if all(d["trajectory_score"] is not None for d in v[lab])]
    print(f"{lab:5s} {st.mean(g):.3f}  (task-mean sd ~ {st.mean(g)/2**0.5/ (2**0.5):.3f})")

def tmean(v, lab):
    s = [d["trajectory_score"] for d in v[lab] if d["trajectory_score"] is not None]
    return st.mean(s) if s else None

print("\n== delta by source / horizon / phase")
for key in ("source", "horizon", "sample_phase"):
    grp = defaultdict(lambda: defaultdict(list))
    for v in rows.values():
        k = v["king"][0][key]; km = tmean(v, "king")
        for lab in ("sft", "dpo"):
            m = tmean(v, lab)
            if m is not None and km is not None: grp[k][lab].append(m - km)
    for k, g in sorted(grp.items(), key=lambda kv: str(kv[0])):
        print(f"{key}={k}: n={len(g['sft'])} sft {st.mean(g['sft']):+.3f}  dpo {st.mean(g['dpo']):+.3f}")

print("\n== behaviour per rollout (generated turns only)")
for lab in FILES:
    th, ac, nthink, turns, sub, subturn, multi = [], [], 0, [], 0, [], 0
    emptythink = 0; nt = 0
    for v in rows.values():
        for d in v[lab]:
            g = gen_turns(d); turns.append(len(g))
            for i, t in enumerate(g):
                c = t["content"]; h, a = think_action(c); nt += 1
                th.append(len(h)); ac.append(len(a))
                if not h.strip(): emptythink += 1
                if len(re.findall(r"```bash", a)) > 1: multi += 1
                if d["submit_marker"] in a:
                    sub += 1; subturn.append(i + 1)
    if not th: th=[0]
    print(f"{lab:5s} gen_turns/rollout={st.mean(turns):.2f} think_chars mean={st.mean(th):.0f} med={st.median(th):.0f} "
          f"p90={q(th,.9):.0f} empty_think={emptythink/nt:.1%} action_chars mean={st.mean(ac):.0f} "
          f"submitted={sub}/200 submit_turn_med={st.median(subturn) if subturn else '-'} multi_bash={multi/nt:.1%}")

print("\n== submitted vs not: score")
for lab in FILES:
    a, b = [], []
    for v in rows.values():
        for d in v[lab]:
            if d["trajectory_score"] is None: continue
            s = any(d["submit_marker"] in think_action(t["content"])[1] for t in gen_turns(d))
            (a if s else b).append(d["trajectory_score"])
    print(f"{lab:5s} submitted n={len(a)} mean={st.mean(a) if a else float('nan'):.3f} | not n={len(b)} mean={st.mean(b) if b else float('nan'):.3f}")

print("\n== sft vs dpo per task correlation of deltas, and where sft loses big")
pairs = []
for sid, v in rows.items():
    km, sm, dm = tmean(v, "king"), tmean(v, "sft"), tmean(v, "dpo")
    if None not in (km, sm, dm): pairs.append((sid, sm - km, dm - km, v["king"][0]["source"], v["king"][0]["sample_phase"]))
xs, ys = [p[1] for p in pairs], [p[2] for p in pairs]
mx, my = st.mean(xs), st.mean(ys)
cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys)); r = cov / (sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**.5
print(f"corr(sft_delta, dpo_delta)={r:.2f}; sft-dpo paired mean={st.mean(x-y for x,y in zip(xs,ys)):+.3f}")
big = sorted(pairs, key=lambda p: p[1])[:10]
for p in big: print(f"  sft {p[1]:+.3f} dpo {p[2]:+.3f} {p[3]} {p[4]} {p[0][-40:]}")
print("share of sft total deficit from worst 10 tasks:", f"{sum(p[1] for p in big)/sum(xs):.0%}")

# paired bootstrap sft - dpo
random.seed(0); diffs = [x - y for x, y in zip(xs, ys)]
bs = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(4000))
print(f"sft-dpo CI95 [{bs[100]:+.3f}, {bs[3899]:+.3f}]")

print("\n== per-question: which question tags does sft miss more (needs per-question verdicts) ->", "questions have verdicts?",
      any("verdict" in qq or "score" in qq for v in rows.values() for qq in v["sft"][0]["questions"][:1]))
