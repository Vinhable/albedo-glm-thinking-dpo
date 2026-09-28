#!/usr/bin/env python3
"""A tiny random Qwen3.5-MoE (same architecture, layer mix, vocabulary and tensor names as the King)
for CPU tests of the trainer. Writes config.json and a `random_init` marker; weights are created
at load time.

    E:/venvs/albedo-cpu/Scripts/python.exe scripts/make_tiny_qwen35moe.py --out E:/albedo-storage-temp/tiny-qwen35moe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GENESIS_CONFIG = Path.home() / (".cache/huggingface/hub/models--dendriteholdings--albedo-qwen3.6-35b-king-genesis/"
                                "snapshots/d7934c55d650e3a73de3081c11ad2c864009f4b7/config.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(GENESIS_CONFIG.read_text(encoding="utf-8"))
    text = config["text_config"]
    text.update({
        "hidden_size": 64, "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 32, "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 16, "linear_value_head_dim": 16, "mtp_num_hidden_layers": 1,
    })
    vision = config["vision_config"]
    vision.update({"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                   "out_hidden_size": 64})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "random_init").write_text("weights are random; for CPU tests only\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
