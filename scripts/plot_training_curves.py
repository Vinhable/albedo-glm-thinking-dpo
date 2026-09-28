#!/usr/bin/env python3
"""Training curves of a train_vinhable_dpo.py run as one standalone SVG (no plotting library needed).

Panels: loss split into its DPO and NLL parts, chosen NLL, chosen/rejected log-ratios (train and dev,
optionally against a comparison run), accuracy, grad norm and LR.

    py -3 scripts/plot_training_curves.py results/training/single-v3/metrics.jsonl \
        --nll-weight 50 --compare results/training/single-v2/metrics.jsonl --out curves.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 520, 260, 60, 16, 34, 38
COLORS = {"a": "#2563eb", "b": "#dc2626", "c": "#16a34a", "d": "#9333ea", "e": "#6b7280"}


def load(path: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    train = [r for r in rows if "grad_norm" in r]
    dev = [r for r in rows if "dev_accuracy" in r]
    return train, dev


def rolling(values: list[float], k: int = 5) -> list[float]:
    return [sum(values[max(0, i - k + 1): i + 1]) / len(values[max(0, i - k + 1): i + 1]) for i in range(len(values))]


def panel(title: str, series: list[tuple[str, str, list[tuple[float, float]], bool]], x0: float, y0: float,
          x_max: float, log_y: bool = False) -> str:
    """series: (label, color, points, dashed)."""
    import math
    pts = [p for _, _, s, _ in series for p in s]
    ys = [math.log10(y) if log_y else y for _, y in pts if not log_y or y > 0]
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-12:
        hi, lo = hi + 1, lo - 1
    margin = (hi - lo) * 0.08
    lo, hi = lo - margin, hi + margin
    iw, ih = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(x): return x0 + PAD_L + iw * x / x_max
    def sy(y):
        v = math.log10(y) if log_y else y
        return y0 + PAD_T + ih * (1 - (v - lo) / (hi - lo))

    out = [f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" fill="#fff" stroke="#e5e7eb"/>',
           f'<text x="{x0 + PAD_L}" y="{y0 + 20}" font-size="13" font-weight="600">{title}</text>']
    for i in range(5):  # y grid
        v = lo + (hi - lo) * i / 4
        y = y0 + PAD_T + ih * (1 - i / 4)
        label = f"{10 ** v:.3g}" if log_y else f"{v:.3g}"
        out.append(f'<line x1="{x0 + PAD_L}" x2="{x0 + W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#f1f5f9"/>')
        out.append(f'<text x="{x0 + PAD_L - 6}" y="{y + 4:.1f}" font-size="10" text-anchor="end" fill="#6b7280">{label}</text>')
    if not log_y and lo < 0 < hi:
        out.append(f'<line x1="{x0 + PAD_L}" x2="{x0 + W - PAD_R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="#9ca3af"/>')
    for i in range(0, int(x_max) + 1, 10):  # x ticks
        out.append(f'<text x="{sx(i):.1f}" y="{y0 + H - PAD_B + 16}" font-size="10" text-anchor="middle" fill="#6b7280">{i}</text>')
    out.append(f'<text x="{x0 + PAD_L + iw / 2}" y="{y0 + H - 6}" font-size="10" text-anchor="middle" fill="#6b7280">step</text>')
    lx = x0 + PAD_L + 8
    for label, color, pts_, dashed in series:
        if not pts_:
            continue
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(pts_)
                     if not log_y or y > 0)
        dash = ' stroke-dasharray="5 3"' if dashed else ""
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>')
        if len(pts_) <= 12:
            out += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{color}"/>' for x, y in pts_]
        out.append(f'<text x="{lx}" y="{y0 + PAD_T + 12}" font-size="10" fill="{color}">{label}</text>')
        lx += 7 * len(label) + 14
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path)
    p.add_argument("--nll-weight", type=float, required=True, help="to split the loss into DPO and NLL parts")
    p.add_argument("--compare", type=Path, help="another run's metrics.jsonl (dev log-ratios drawn dashed)")
    p.add_argument("--compare-label", default="v2")
    p.add_argument("--title", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    train, dev = load(args.metrics)
    steps = [r["step"] for r in train]
    x_max = max(steps + [r["step"] for r in dev])
    loss = [r["loss"] for r in train]
    nll_part = [args.nll_weight * r["chosen_nll"] for r in train]
    dpo_part = [l - n for l, n in zip(loss, nll_part)]
    z = lambda xs, ys: list(zip(xs, ys))

    panels = [
        ("Loss (train) = DPO part + NLL part", [
            ("total", COLORS["a"], z(steps, loss), False),
            (f"NLL part ({args.nll_weight:g}×nll)", COLORS["c"], z(steps, nll_part), False),
            ("DPO part", COLORS["b"], z(steps, dpo_part), False),
            ("total, 5-step mean", COLORS["e"], z(steps, rolling(loss)), True)], False),
        ("Chosen NLL per token (train, GLM text)", [
            ("chosen_nll", COLORS["c"], z(steps, [r["chosen_nll"] for r in train]), False),
            ("5-step mean", COLORS["e"], z(steps, rolling([r["chosen_nll"] for r in train])), True)], False),
        ("Log-ratio vs King, nat/token", [
            ("train chosen (5-step mean)", COLORS["a"], z(steps, rolling([r["chosen_logratio"] for r in train])), False),
            ("train rejected (5-step mean)", COLORS["b"], z(steps, rolling([r["rejected_logratio"] for r in train])), False),
            ("dev chosen", COLORS["a"], [(r["step"], r["dev_chosen_logratio"]) for r in dev], True),
            ("dev rejected", COLORS["b"], [(r["step"], r["dev_rejected_logratio"]) for r in dev], True)], False),
        ("Accuracy (margin > 0)", [
            ("train, 5-step mean", COLORS["d"], z(steps, rolling([r["accuracy"] for r in train])), False),
            ("dev (91 rows)", COLORS["a"], [(r["step"], r["dev_accuracy"]) for r in dev], True)], False),
        ("Grad norm before clipping (clip = 1.0), log scale", [
            ("grad_norm", COLORS["b"], z(steps, [r["grad_norm"] for r in train]), False)], True),
        ("Learning rate", [("lr", COLORS["e"], z(steps, [r["lr"] for r in train]), False)], False),
    ]
    if args.compare:
        _, dev2 = load(args.compare)
        panels[2][1].extend([
            (f"{args.compare_label} dev chosen", "#93c5fd", [(r["step"], r["dev_chosen_logratio"]) for r in dev2], True),
            (f"{args.compare_label} dev rejected", "#fca5a5", [(r["step"], r["dev_rejected_logratio"]) for r in dev2], True)])

    cols, gap = 2, 12
    rows_n = (len(panels) + cols - 1) // cols
    total_w, total_h = cols * W + (cols + 1) * gap, rows_n * H + (rows_n + 1) * gap + 30
    body = []
    for i, (title, series, log_y) in enumerate(panels):
        x0 = gap + (i % cols) * (W + gap)
        y0 = 30 + gap + (i // cols) * (H + gap)
        body.append(panel(title, series, x0, y0, x_max, log_y))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
           f'font-family="system-ui, sans-serif">\n<rect width="100%" height="100%" fill="#f8fafc"/>\n'
           f'<text x="{gap}" y="24" font-size="15" font-weight="700">{args.title}</text>\n' + "\n".join(body) + "\n</svg>\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    print(f"wrote {args.out} ({len(train)} train steps, {len(dev)} dev evals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
