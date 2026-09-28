#!/usr/bin/env python3
"""Production-exact training sequences for `vinhable_single` / `vinhable_double` rows.

A row (one behaviour-group prefix `k` of a pair) supervises only group `k`'s assistant turns. Each
supervised turn becomes its **own sequence**, rendered the way production prompted that turn:

    chat_template(prompt + every earlier message of that side, add_generation_prompt=True,
                  enable_thinking=True)            # ends with <|im_start|>assistant\\n<think>\\n
    + the turn's content (reasoning, </think>, action) + <|im_end|>

Earlier turns go through the canonical template verbatim, and the template drops their reasoning
exactly as production's history rendering does, so no supervised turn ever sees past thinking
(rendering a whole side as one sequence would show it). Loss covers the turn's content and
`<|im_end|>`: reasoning, `</think>` and action all train; `<think>\\n` is part of the prompt.

No torch here: the trainer and the CPU checks share this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT / "assets" / "tokenizers" / "Qwen3.6-35B-A3B"
GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass
class Sequence:
    """One supervised turn: `prompt_ids` then `target_ids` (the turn + <|im_end|>)."""

    prompt_ids: list[int]
    target_ids: list[int]

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.target_ids)


@dataclass
class RowExample:
    uid: str
    pair_id: str
    split: str
    direction: str
    chosen: list[Sequence] = field(default_factory=list)
    rejected: list[Sequence] = field(default_factory=list)

    def tokens(self, side: str) -> int:
        return sum(len(s.target_ids) for s in getattr(self, side))

    @property
    def max_length(self) -> int:
        return max(s.length for s in self.chosen + self.rejected)

    @property
    def total_length(self) -> int:
        return sum(s.length for s in self.chosen + self.rejected)


class Renderer:
    """Canonical Qwen3.6 chat template (== genesis) + tokenizer, without transformers."""

    def __init__(self, tokenizer_dir: Path = TOKENIZER_DIR) -> None:
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        def raise_exception(message: str) -> None:
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string((tokenizer_dir / "chat_template.jinja").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        self.im_end = self.tokenizer.token_to_id("<|im_end|>")
        self.think_close = self.tokenizer.token_to_id("</think>")
        if self.im_end is None or self.think_close is None:
            raise ValueError("tokenizer lacks <|im_end|> or </think>")

    def prompt(self, messages: list[dict[str, Any]]) -> str:
        text = self.template.render(messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                                    add_generation_prompt=True, enable_thinking=True)
        if not text.endswith(GENERATION_PROMPT):
            raise ValueError("rendered prompt does not end with the thinking generation prompt")
        return text

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def sequence(self, context: list[dict[str, Any]], content: str) -> Sequence:
        return Sequence(prompt_ids=self.encode(self.prompt(context)),
                        target_ids=self.encode(content) + [self.im_end])


def supervised(message: dict[str, Any]) -> bool:
    """A turn trains only if it is marked for loss and closes its think block: learning a turn that
    never closes `</think>` is how the lab's idea-35 export stopped closing it (0.8%)."""
    return message["role"] == "assistant" and bool(message.get("loss")) and "</think>" in message["content"]


def side_sequences(renderer: Renderer, prompt: list[dict[str, Any]], side: list[dict[str, Any]]) -> list[Sequence]:
    context = [{"role": m["role"], "content": m["content"]} for m in prompt]
    out = []
    for index, message in enumerate(side):
        if supervised(message):
            history = [{"role": m["role"], "content": m["content"]} for m in side[:index]]
            out.append(renderer.sequence(context + history, message["content"]))
    return out


def row_example(renderer: Renderer, row: dict[str, Any]) -> RowExample:
    return RowExample(
        uid=row["sample_uid"], pair_id=row["pair_id"], split=row["split"], direction=row["direction"],
        chosen=side_sequences(renderer, row["prompt"], row["chosen"]),
        rejected=side_sequences(renderer, row["prompt"], row["rejected"]),
    )


def load_rows(data_dir: Path, split: str) -> list[dict[str, Any]]:
    rows = list(read_jsonl(data_dir / f"{split}.jsonl"))
    uids = [r["sample_uid"] for r in rows]
    if len(uids) != len(set(uids)):  # the lab once keyed reference log-probs on a non-unique id
        raise ValueError(f"{split}: sample_uid is not unique")
    return rows


def char_length(row: dict[str, Any]) -> int:
    """Cheap proxy of a row's cost for batching (no tokenisation)."""
    prompt = sum(len(m["content"]) for m in row["prompt"])
    total = 0
    for side in ("chosen", "rejected"):
        running = prompt
        for m in row[side]:
            if supervised(m):
                total += running + len(m["content"])
            running += len(m["content"])
    return total


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def batches(rows: Iterable[dict[str, Any]], per_step: int, seed: int, epoch: int) -> list[list[dict[str, Any]]]:
    """Shuffle by a seeded hash, then keep rows of similar cost together inside each step."""
    ordered = sorted(rows, key=lambda r: stable_hash(f"{seed}:{epoch}:{r['sample_uid']}"))
    steps = [ordered[i:i + per_step] for i in range(0, len(ordered), per_step)]
    return [sorted(step, key=char_length, reverse=True) for step in steps]
