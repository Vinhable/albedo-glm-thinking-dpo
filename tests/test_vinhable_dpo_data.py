"""Rendering contract of the vinhable DPO data layer (CPU, no torch)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vinhable_dpo_data import GENERATION_PROMPT, Renderer, row_example, supervised  # noqa: E402

PROMPT = [{"role": "system", "content": "You are an agent.", "loss": False},
          {"role": "user", "content": "Fix the bug in a.py.", "loss": False}]
T1 = "SECRET-REASONING-ONE about the layout\n</think>\n\nTHOUGHT: list files\n\n```bash\nls\n```"
T2 = "SECRET-REASONING-TWO about a.py\n</think>\n\n```bash\ncat a.py\n```"
UNCLOSED = "reasoning that never closes and runs into ```bash\nls\n```"


def row(side_turns):
    side = []
    for content, loss in side_turns:
        side.append({"role": "assistant", "content": content, "loss": loss})
        side.append({"role": "user", "content": "<returncode>0</returncode>\n<output>\nok\n</output>", "loss": False})
    return {"sample_uid": "p#k1", "pair_id": "p", "split": "train", "direction": "glm_over_king",
            "prompt": PROMPT, "chosen": side, "rejected": side}


def test_each_supervised_turn_is_its_own_sequence_without_past_reasoning():
    r = Renderer()
    ex = row_example(r, row([(T1, False), (T2, True)]))
    assert len(ex.chosen) == 1
    seq = ex.chosen[0]
    prompt = r.tokenizer.decode(seq.prompt_ids, skip_special_tokens=False)
    assert prompt.endswith(GENERATION_PROMPT)
    assert "SECRET-REASONING-ONE" not in prompt          # earlier reasoning dropped, as production does
    assert "THOUGHT: list files" in prompt                # earlier action kept
    assert r.tokenizer.decode(seq.target_ids, skip_special_tokens=False) == T2 + "<|im_end|>"
    assert r.think_close in seq.target_ids


def test_two_supervised_turns_give_two_sequences_in_order():
    r = Renderer()
    ex = row_example(r, row([(T1, True), (T2, True)]))
    assert [r.tokenizer.decode(s.target_ids, skip_special_tokens=False) for s in ex.chosen] == \
        [T1 + "<|im_end|>", T2 + "<|im_end|>"]
    assert len(ex.chosen[1].prompt_ids) > len(ex.chosen[0].prompt_ids)


def test_turn_that_never_closes_think_is_not_supervised():
    assert not supervised({"role": "assistant", "content": UNCLOSED, "loss": True})
    assert supervised({"role": "assistant", "content": T1, "loss": True})
    assert not supervised({"role": "assistant", "content": T1, "loss": False})
    ex = row_example(Renderer(), row([(UNCLOSED, True), (T2, True)]))
    assert len(ex.chosen) == 1
