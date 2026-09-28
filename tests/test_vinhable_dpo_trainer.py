"""CPU checks of the vinhable DPO trainer's maths on a tiny random Qwen3.5-MoE.

Needs torch + transformers (the E: venv): E:/venvs/albedo-cpu/Scripts/python.exe -m pytest -q tests/test_vinhable_dpo_trainer.py
The tiny model is made by scripts/make_tiny_qwen35moe.py; the rows by the CPU-test data folder.
"""

import argparse
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_vinhable_dpo import TRAINABLE, dpo_terms, make_sequence_module, pack, run_batch, sequence_logp  # noqa: E402
from vinhable_dpo_data import Renderer, Sequence  # noqa: E402

TINY = Path("E:/albedo-storage-temp/tiny-qwen35moe")


def args(**kw):
    base = dict(beta=0.1, agg="mean", norm_tokens=512.0, label_smoothing=0.1, nll_weight=0.1)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("agg", ["mean", "sum"])
def test_dpo_derivatives_match_autograd(agg):
    a = args(agg=agg)
    pc, pr, rc, rr, nc, nr = -120.0, -95.0, -118.0, -99.0, 40, 30
    t = dpo_terms(pc, pr, rc, rr, nc, nr, a)
    x = torch.tensor([pc, pr], dtype=torch.float64, requires_grad=True)
    if agg == "mean":
        h = a.beta * a.norm_tokens * ((x[0] - rc) / nc - (x[1] - rr) / nr)
    else:
        h = a.beta * ((x[0] - rc) - (x[1] - rr))
    ls = a.label_smoothing
    loss = -(1 - ls) * torch.nn.functional.logsigmoid(h) - ls * torch.nn.functional.logsigmoid(-h) \
        + a.nll_weight * (-x[0] / nc)
    loss.backward()
    assert math.isclose(t["loss"], float(loss), rel_tol=1e-9)
    assert math.isclose(t["coef_chosen"], float(x.grad[0]), rel_tol=1e-9)
    assert math.isclose(t["coef_rejected"], float(x.grad[1]), rel_tol=1e-9)


@pytest.mark.skipif(not TINY.exists(), reason="run scripts/make_tiny_qwen35moe.py first")
def test_two_pass_gradient_equals_direct_autograd():
    """Pass A (no grad) + coefficient-scaled pass B gives the same gradient as one graph."""
    from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration

    torch.manual_seed(0)
    model = Qwen3_5MoeForConditionalGeneration(AutoConfig.from_pretrained(TINY)).to(torch.float32)
    for name, p in model.named_parameters():
        p.requires_grad = bool(TRAINABLE.match(name))
    runner = make_sequence_module(model)
    renderer = Renderer()
    make = lambda u, a: Sequence(renderer.encode(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
                                 renderer.encode(a) + [renderer.im_end])
    chosen = [make("fix the bug", "look first\n</think>\n\n```bash\nls\n```"),
              make("then?", "read it\n</think>\n\n```bash\ncat a.py\n```")]
    rejected = [make("fix the bug", "guess\n</think>\n\n```bash\nrm -rf x\n```")]
    ref = {"chosen": -300.0, "rejected": -150.0}
    nc, nr = sum(len(s.target_ids) for s in chosen), sum(len(s.target_ids) for s in rejected)
    a = args()
    params = [p for p in model.parameters() if p.requires_grad]

    # direct: one graph over every sequence
    pc = sum(sequence_logp(runner, s, "cpu") for s in chosen)
    pr = sum(sequence_logp(runner, s, "cpu") for s in rejected)
    h = a.beta * a.norm_tokens * ((pc - ref["chosen"]) / nc - (pr - ref["rejected"]) / nr)
    ls = a.label_smoothing
    loss = -(1 - ls) * torch.nn.functional.logsigmoid(h) - ls * torch.nn.functional.logsigmoid(-h) \
        + a.nll_weight * (-pc / nc)
    direct = torch.autograd.grad(loss, params)

    # two passes, as the trainer does
    with torch.no_grad():
        sc = sum(float(sequence_logp(runner, s, "cpu")) for s in chosen)
        sr = sum(float(sequence_logp(runner, s, "cpu")) for s in rejected)
    t = dpo_terms(sc, sr, ref["chosen"], ref["rejected"], nc, nr, a)
    for p in params:
        p.grad = None
    for coef, seqs in ((t["coef_chosen"], chosen), (t["coef_rejected"], rejected)):
        for s in seqs:
            (coef * sequence_logp(runner, s, "cpu")).backward()
    worst = max(float((g - p.grad).abs().max() / (g.abs().max() + 1e-12)) for g, p in zip(direct, params))
    assert math.isclose(t["loss"], float(loss), rel_tol=1e-5)
    assert worst < 1e-4, worst


@pytest.mark.skipif(not TINY.exists(), reason="run scripts/make_tiny_qwen35moe.py first")
def test_right_padded_batch_matches_single_sequences():
    """Packing sequences into one right-padded forward must not change any sequence's log-prob."""
    from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration

    torch.manual_seed(0)
    model = Qwen3_5MoeForConditionalGeneration(AutoConfig.from_pretrained(TINY)).to(torch.float32).eval()
    runner = make_sequence_module(model)
    r = Renderer()
    head = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    seqs = [Sequence(r.encode(head.format("x " * n)),
                     r.encode("ok\n</think>\n\n```bash\nls\n```") + [r.im_end]) for n in (3, 40, 11)]
    with torch.no_grad():
        single = [float(sequence_logp(runner, s, "cpu")) for s in seqs]
        batched = run_batch(runner, seqs, "cpu").tolist()
    assert all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-4) for a, b in zip(single, batched)), (single, batched)


def test_pack_respects_budgets():
    items = [(n, t) for n, t in ((100, 10), (90, 10), (50, 60), (40, 5), (10, 1))]
    batches = pack(items, lambda i: i[0], max_tokens=200, max_targets=70, targets_of=lambda i: i[1])
    for b in batches:
        assert max(i[0] for i in b) * len(b) <= 200 or len(b) == 1
        assert sum(i[1] for i in b) <= 70 or len(b) == 1
    assert sorted(i for b in batches for i in b) == sorted(items)
