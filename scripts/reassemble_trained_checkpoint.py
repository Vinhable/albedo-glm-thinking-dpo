#!/usr/bin/env python3
"""Full checkpoint = the King's checkpoint with the trained tensors of a slim export swapped in.

The trainer exports only the tensors it trained (`export-step<N>/trained.safetensors`, names as in
the King's checkpoint, bf16). This rewrites the King's shards that hold any of them and copies every
other file byte for byte (config, tokenizer, chat template, generation config, processors), so the
result has the King's 1,045 tensor names, shapes and dtypes and its metadata files unchanged, which
is what Albedo's admission check compares. Writes `<out>.reassembly.json` beside the folder (per-file
sha256), so the folder holds exactly the King's file set.

    python scripts/reassemble_trained_checkpoint.py --king /workspace/models/king127 \
        --trained /workspace/runs/single/export-step40/trained.safetensors --out /workspace/models/single-step40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--king", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    index = json.loads((args.king / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    trained = load_file(str(args.trained))
    unknown = sorted(set(trained) - set(weight_map))
    if unknown:
        raise SystemExit(f"{len(unknown)} trained tensors are not in the King's checkpoint, e.g. {unknown[:3]}")
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in trained:
        by_shard[weight_map[name]].append(name)

    args.out.mkdir(parents=True, exist_ok=False)
    report = {"king": str(args.king), "trained": str(args.trained), "replaced_tensors": len(trained),
              "rewritten_shards": sorted(by_shard), "files": {}}
    for path in sorted(p for p in args.king.iterdir() if p.is_file()):
        target = args.out / path.name
        if path.name in by_shard:
            tensors = load_file(str(path))
            for name in by_shard[path.name]:
                old, new = tensors[name], trained[name]
                if old.shape != new.shape or old.dtype != new.dtype:
                    raise SystemExit(f"{name}: {tuple(new.shape)}/{new.dtype} != King {tuple(old.shape)}/{old.dtype}")
                tensors[name] = new.contiguous()
            with safe_open(str(path), framework="pt") as handle:
                metadata = handle.metadata()
            save_file(tensors, str(target), metadata=metadata)
        else:
            shutil.copy2(path, target)
        report["files"][path.name] = {"sha256": sha256(target), "king_sha256": sha256(path),
                                      "rewritten": path.name in by_shard}
    # the result must list exactly the King's tensors with the King's shapes and dtypes
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(args.out / shard), framework="pt") as new, safe_open(str(args.king / shard), framework="pt") as old:
            if set(new.keys()) != set(old.keys()):
                raise SystemExit(f"{shard}: tensor names differ from the King")
            for name in old.keys():
                a, b = new.get_slice(name), old.get_slice(name)
                if a.get_shape() != b.get_shape() or a.get_dtype() != b.get_dtype():
                    raise SystemExit(f"{name}: shape/dtype differs from the King")
    unchanged_meta = [n for n, f in report["files"].items() if not n.endswith(".safetensors") and f["sha256"] != f["king_sha256"]]
    if unchanged_meta:
        raise SystemExit(f"metadata files changed: {unchanged_meta}")
    report["tensors"] = len(weight_map)
    # beside the folder, not in it: the checkpoint must hold exactly the King's files
    (args.out.parent / f"{args.out.name}.reassembly.json").write_text(json.dumps(report, indent=1) + "\n",
                                                                      encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("replaced_tensors", "rewritten_shards", "tensors")}, indent=1))


if __name__ == "__main__":
    main()
