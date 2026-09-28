"""Count tasks where the King scored low and the GLM-5.2 reference scored high.

Reads the `glm_5_2` split of HF `dendriteholdings/albedo` (downloaded locally) and reports, per
threshold, how many samples have King low / GLM high.

Scales differ: the reference's own `score` is the weighted share of checklist items it earned
(`earned` is a boolean in every row), while `king_score` is the King's production score. An eval is
counted as graded-era when some King score sits strictly between 0.99 and 1 (a logprob expectation;
a three-vote binary mean over ~17 items cannot land there).

    py -3 scripts/analyze_glm_reference_headroom.py --root E:/albedo-storage-temp/hf-albedo-reference-20260927
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

HELD_OUT_EVAL_PREFIXES = ("719dbe80", "3a48801e")  # the lab's duel sets


def assistant_turns(messages) -> int:
    return sum(1 for m in messages or [] if m.get("role") == "assistant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    files = sorted((args.root / "data" / "glm_5_2").glob("*.parquet"))
    rows = []
    for path in files:
        table = pq.read_table(path, columns=[
            "sample_id", "run", "made_edit", "score", "questions", "king_score", "eval_run_id", "messages"])
        for r in table.to_pylist():
            r["turns"] = assistant_turns(r.pop("messages"))
            r["file"] = path.name
            rows.append(r)
    print(f"files {len(files)}  rows {len(rows)}  evals {len({r['eval_run_id'] for r in rows})}")

    # scale per eval, read from the King's scores (see the module docstring)
    eval_scale: dict[str, str] = {}
    for r in rows:
        if r["score"] is None or r["king_score"] is None:
            continue
        graded = 0.99 < float(r["king_score"]) < 1.0
        if graded or r["eval_run_id"] not in eval_scale:
            eval_scale[r["eval_run_id"]] = "graded" if graded else eval_scale.get(r["eval_run_id"], "binary")
    print("scored evals by King scale:", Counter(eval_scale.values()))

    # one record per (eval, sample)
    samples: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["eval_run_id"], r["sample_id"])
        s = samples.setdefault(key, {"eval": r["eval_run_id"], "sample_id": r["sample_id"], "king": r["king_score"],
                                     "glm": [], "edit": [], "turns": [], "scale": eval_scale.get(r["eval_run_id"])})
        if r["score"] is not None:
            s["glm"].append(float(r["score"]))
            s["edit"].append(r["made_edit"])
            s["turns"].append(r["turns"])
        if s["king"] is None and r["king_score"] is not None:
            s["king"] = r["king_score"]

    scored = [s for s in samples.values() if s["glm"] and s["king"] is not None]
    binary = [s for s in scored if s["scale"] != "graded"]
    print(f"binary-era scored samples (reported for reference, not in the pool): {len(binary)}")
    graded = [s for s in scored if s["scale"] == "graded"]
    held = [s for s in graded if s["eval"].startswith(HELD_OUT_EVAL_PREFIXES)]
    pool = [s for s in graded if not s["eval"].startswith(HELD_OUT_EVAL_PREFIXES)]
    print(f"samples {len(samples)}  with GLM+King score {len(scored)}  graded {len(graded)}  "
          f"(held-out duel sets {len(held)}, pool {len(pool)})")
    print(f"graded evals {len({s['eval'] for s in graded})}; GLM runs per sample {Counter(len(s['glm']) for s in pool)}")

    kings = [float(s["king"]) for s in pool]
    print("\nKing score distribution (pool):")
    for lo, hi, label in [(1.0, 1.01, "= 1"), (0.9, 1.0, "[0.9,1)"), (0.7, 0.9, "[0.7,0.9)"),
                          (0.5, 0.7, "[0.5,0.7)"), (0.0001, 0.5, "(0,0.5)"), (-1, 0.0001, "= 0")]:
        n = sum(lo <= k < hi for k in kings)
        print(f"  King {label:10s} {n:6d}  {n / len(kings):.1%}")
    print(f"  King mean {st.mean(kings):.3f}  GLM best mean {st.mean(max(s['glm']) for s in pool):.3f}")

    def best(s):  # best GLM run that made an edit when any did
        edited = [g for g, e in zip(s["glm"], s["edit"]) if e]
        return max(edited) if edited else max(s["glm"])

    print("\nKing low x GLM high (pool, best GLM run; unique tasks = distinct sample_id):")
    print(f"  {'King <=':>8s} | " + " | ".join(f"GLM>={g:.1f}" for g in (0.7, 0.8, 0.9)) + " | gap>=0.2 | gap>=0.3")
    table = {}
    for kmax in (0.3, 0.5, 0.7, 0.8, 0.9, 0.99):
        cells = []
        for gmin in (0.7, 0.8, 0.9):
            sel = [s for s in pool if float(s["king"]) <= kmax and best(s) >= gmin]
            cells.append(f"{len(sel):5d}/{len({s['sample_id'] for s in sel}):5d}")
            table[f"king<={kmax}|glm>={gmin}"] = len(sel)
        for gap in (0.2, 0.3):
            sel = [s for s in pool if float(s["king"]) <= kmax and best(s) - float(s["king"]) >= gap]
            cells.append(f"{len(sel):5d}")
            table[f"king<={kmax}|gap>={gap}"] = len(sel)
        print(f"  {kmax:8.2f} | " + " | ".join(cells))

    focus = [s for s in pool if float(s["king"]) <= 0.7 and best(s) >= 0.9]
    if focus:
        src = Counter(s["sample_id"].split("/")[0] for s in focus)
        zero_king = sum(float(s["king"]) == 0 for s in focus)
        consistent = sum(min(s["glm"]) >= 0.7 for s in focus)
        print(f"\nFocus King<=0.7 & GLM>=0.9: {len(focus)} samples, {len({s['sample_id'] for s in focus})} tasks")
        print(f"  sources {dict(src)}")
        print(f"  King exactly 0: {zero_king}  |  every GLM run >= 0.7: {consistent}  |  "
              f"GLM runs per sample {dict(Counter(len(s['glm']) for s in focus))}")
        print(f"  GLM assistant turns median {st.median(t for s in focus for t in s['turns'])}")

    if args.out:
        args.out.write_text(json.dumps({
            "rows": len(rows), "samples": len(samples), "graded_samples": len(graded),
            "held_out_samples": len(held), "pool_samples": len(pool), "table": table,
            "graded_evals": sorted({s["eval"] for s in pool}),
            "held_out_evals": sorted({s["eval"] for s in held}),
            "focus": [{k: s[k] for k in ("eval", "sample_id", "king", "glm", "edit", "turns")} for s in focus],
        }, indent=1), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
