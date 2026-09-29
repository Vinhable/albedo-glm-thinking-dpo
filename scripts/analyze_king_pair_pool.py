"""How much King-CXXVII self-improvement data the graded production crawl holds (no API).

Unit: one (eval run, sample) with both King rollouts scored by the graded judge. The challenger's two
rollouts of the same sample are scored on the same checklist by the same judge, so they are a fair
"a similar model can do this" signal. GLM reference scores are NOT used to define groups: the
checklist is built from the reference runs and pruned to questions some reference run earned
(judge_api._prune_by_reference), so the best reference run scores ~1 by construction.

Groups (King best = max of the two King rollouts):
  A  King capable       King best >= --capable (0.85)
  B  King inconsistent  not A, |r1 - r2| >= --margin (0.2)       -> best-vs-worst King pairs fix it
  C  King weak          not A, not B                              -> needs outside knowledge (hints)
     C1  a challenger rollout beat King best by >= --margin (a King-like model did it)
     C2  nobody in the run did

Also: King best-vs-worst pair counts at several margins (direction 1), the pool-mean headroom of
best-of-2 / best-of-4, and every count again without the samples already used (duel set, GLM batches).

    python scripts/analyze_king_pair_pool.py --out E:/albedo-storage-temp/king-pair-pool-20260929.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gen_vinhable import rollouts, run_index  # noqa: E402

CRAWL = Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927")
USED = [  # samples already spent on training data or on the duel
    Path("E:/albedo-storage-temp/glm-thinking-batch300-20260927/pilot-input.jsonl"),
    *sorted(Path("E:/albedo-storage-temp").glob("glm-thinking-pilot*-20260927/pilot-input.jsonl")),
    Path("E:/albedo-storage-temp/duel-sft-dpo-vs-king-20260928/duel-input.jsonl"),
]


def used_samples() -> set[str]:
    out: set[str] = set()
    for path in USED:
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                out.update(json.loads(line)["sample_id"] for line in handle if line.strip())
    return out


def load(crawl: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    units = []
    for run in run_index([crawl]):
        for sid, info, found in rollouts(run, counts):
            if info.get("scoring_mode") != "graded_20":
                counts["skip:not_graded"] += 1
                continue
            king = [r["score"] for r in sorted(found, key=lambda r: r["replica"]) if r["side"] == "king"]
            chal = [r["score"] for r in found if r["side"] == "challenger"]
            if len(king) < 2:
                counts["skip:fewer_than_2_king"] += 1
                continue
            units.append({"run": run["eval_run_id"], "sample_id": sid, "phase": info.get("sample_phase"),
                          "source": sid.split("/", 1)[0], "king": king[:2], "chal": chal,
                          "king_version": run["king_version"]})
    return units, counts


def group(u: dict[str, Any], capable: float, margin: float) -> str:
    best, worst = max(u["king"]), min(u["king"])
    if best >= capable:
        return "A"
    if best - worst >= margin:
        return "B"
    return "C1" if u["chal"] and max(u["chal"]) - best >= margin else "C2"


def report(units: list[dict[str, Any]], capable: float, margin: float) -> dict[str, Any]:
    n = len(units)
    groups = Counter(group(u, capable, margin) for u in units)
    kmean = [st.mean(u["king"]) for u in units]
    best2 = [max(u["king"]) for u in units]
    best4 = [max(u["king"] + u["chal"]) for u in units]
    gaps = [max(u["king"]) - min(u["king"]) for u in units]
    out: dict[str, Any] = {
        "units": n,
        "unique_samples": len({u["sample_id"] for u in units}),
        "king_mean": round(st.mean(kmean), 4),
        "king_best_of_2": round(st.mean(best2), 4),
        "best_of_4_with_challenger": round(st.mean(best4), 4),
        "king_pairs_by_margin": {f">={m}": sum(g >= m for g in gaps) for m in (0.1, 0.15, 0.2, 0.3, 0.4)},
        "groups": {g: {"units": groups[g], "share": round(groups[g] / n, 3)} for g in ("A", "B", "C1", "C2")},
    }
    # pool-mean headroom each group could add if its gap were closed (King mean -> best available)
    for g in ("A", "B", "C1", "C2"):
        part = [u for u in units if group(u, capable, margin) == g]
        if not part:
            continue
        out["groups"][g].update({
            "king_mean": round(st.mean(st.mean(u["king"]) for u in part), 3),
            "king_best": round(st.mean(max(u["king"]) for u in part), 3),
            "best_of_4": round(st.mean(max(u["king"] + u["chal"]) for u in part), 3),
            "pool_headroom_to_king_best": round(sum(max(u["king"]) - st.mean(u["king"]) for u in part) / n, 4),
            "pool_headroom_to_best_of_4": round(sum(max(u["king"] + u["chal"]) - st.mean(u["king"]) for u in part) / n, 4),
        })
    for key in ("phase", "source"):
        table: dict[str, Counter[str]] = defaultdict(Counter)
        for u in units:
            table[u[key]][group(u, capable, margin)] += 1
        out[f"groups_by_{key}"] = {k: dict(v) | {"total": sum(v.values())} for k, v in sorted(table.items(), key=lambda kv: str(kv[0]))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crawl", type=Path, default=CRAWL)
    parser.add_argument("--capable", type=float, default=0.85)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    units, counts = load(args.crawl)
    used = used_samples()
    fresh = [u for u in units if u["sample_id"] not in used]
    result = {
        "crawl": str(args.crawl), "king_versions": dict(Counter(u["king_version"] for u in units)),
        "thresholds": {"capable": args.capable, "margin": args.margin}, "load_counts": dict(counts),
        "used_samples": len(used), "all": report(units, args.capable, args.margin),
        "unused": report(fresh, args.capable, args.margin),
    }
    text = json.dumps(result, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
