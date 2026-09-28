"""Read a side's document from a `generated-samples.jsonl` row, in either artifact format.

Older eval artifacts stored each side twice: the turns (`previous_king_turns`,
`challenger_turns`) and the flat judge-facing text (`previous_king_output`,
`challenger_output`). Newer artifacts keep only the turns. The text is rebuilt with upstream's
`scored_output`, the same function the evaluator uses to build what the judge reads; on 2,820
rows of older artifacts the rebuilt text is byte-identical to the stored copy.

Scripts that read the flat field directly silently see an empty string on newer artifacts, which
makes every King rollout look missing or empty. Read through `side_output` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIELDS = {
    "previous_king": ("previous_king_output", "previous_king_turns"),
    "challenger": ("challenger_output", "challenger_turns"),
}


def side_output(sample: dict[str, Any], side: str) -> str:
    """The judge-facing text of one side: the stored copy if present, else rebuilt from turns."""
    output_key, turns_key = FIELDS[side]
    stored = sample.get(output_key)
    if stored:
        return str(stored)
    turns = sample.get(turns_key)
    if not turns:
        return ""
    # Imported lazily so scripts reading older artifacts never pay for the eval-service import.
    from albedo_eval_service.remote.generation import scored_output

    return scored_output(turns)


def backfill_side_outputs(sample: dict[str, Any]) -> dict[str, Any]:
    """Fill in missing flat fields in place, for code that forwards whole rows."""
    for side, (output_key, _) in FIELDS.items():
        if not sample.get(output_key):
            text = side_output(sample, side)
            if text:
                sample[output_key] = text
    return sample
