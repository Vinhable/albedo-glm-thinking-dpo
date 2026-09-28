#!/usr/bin/env python3
"""Local proxy of the Albedo dedup gate: would this checkpoint be rejected against the King?

Runs upstream's own `fingerprint` and `decide` (src/model_validation/dedup) on the candidate and on
the King, with a locally generated secret instead of the validator's. The secret only fixes the
random sketch projections, so the statistics the verdict rests on (rel_struct against
DEDUP_REL_TRIVIAL, sampled-weight density against DEDUP_DENS_MIN, spectral F_struct) come out on
the same scale as the real gate, though not bit-identical. Treat a margin under ~20% of a
threshold as a fail.

Needs torch and a GPU; reads both checkpoints tensor by tensor.

    py -3 scripts/check_dedup_gate_local.py --candidate out/rehydrated --king /models/king-CXXVII \
        --seed-model /models/genesis --report out/gate-local.json
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True, help="rehydrated full checkpoint dir")
    parser.add_argument("--king", type=Path, required=True, help="King checkpoint dir")
    parser.add_argument("--seed-model", type=Path, required=True,
                        help="genesis checkpoint dir (the gate canonicalises against it)")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="cpu: leave the GPUs to a running job (the sketch statistics are ratios)")
    args = parser.parse_args()

    import torch

    from model_validation.dedup.signals import mats
    from model_validation.dedup.sketch import fingerprint
    from model_validation.dedup.verdict import Thresholds, decide

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    secret = secrets.token_bytes(32)  # never persisted: a fresh projection each check
    king_doc = fingerprint(str(args.king), str(args.seed_model), secret, device, model_uri="king")
    cand_doc = fingerprint(str(args.candidate), str(args.seed_model), secret, device, model_uri="candidate")
    thresholds = Thresholds.from_settings()
    verdict = decide(mats(cand_doc), {"king": mats(king_doc)}, None, cand_doc["identity_frac"], thresholds)
    metrics = verdict.metrics
    report = {
        "verdict": verdict.status,
        "reason": verdict.reason,
        "message": verdict.message,
        "notes": verdict.notes,
        "rel_struct": metrics.get("rel_struct"),
        "rel_trivial_threshold": thresholds.rel_trivial,
        "density": metrics.get("density"),
        "density_threshold": thresholds.dens_min,
        "F_struct": metrics.get("F"),
        "F_noise_threshold": thresholds.f_noise,
        "embed_ratio": metrics.get("embed_ratio"),
        "distances": metrics.get("distances"),
        "tensors_identical_hash": cand_doc["tensors_hash"] == king_doc["tensors_hash"],
        "caveat": "local secret; same statistics as the validator, not bit-identical",
    }
    headroom = (report["rel_struct"] or 0) / thresholds.rel_trivial
    report["rel_struct_headroom_x"] = round(headroom, 2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if verdict.status == "PASS" and headroom >= 1.2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
