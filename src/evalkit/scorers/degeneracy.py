"""Detect degenerate generation that aggregate accuracy metrics cannot see.

A model can produce a correct first answer and then keep going: emitting the
same object repeatedly, reasoning aloud after the answer, or looping until it
hits the token cap. The extractor recovers the first valid object, so accuracy
looks fine while the model is actually misbehaving.

This costs real money in production (tokens billed for nothing), real latency
(the request runs to the cap), and signals instability. It deserves its own
metric rather than being averaged away.
"""

import json
import re
from dataclasses import dataclass

_FENCE_LINE = re.compile(r"```(?:json)?", re.IGNORECASE)


@dataclass
class Degeneracy:
    hit_token_cap: bool = False
    n_json_objects: int = 0
    self_contradictory: bool = False   # multiple objects that disagree
    wasted_tokens: int = 0             # tokens generated after the first object

    @property
    def multiple_objects(self) -> bool:
        return self.n_json_objects > 1

    @property
    def is_degenerate(self) -> bool:
        return self.hit_token_cap or self.multiple_objects

    @property
    def reasons(self) -> list[str]:
        out = []
        if self.hit_token_cap:
            out.append("hit token cap")
        if self.multiple_objects:
            out.append(f"{self.n_json_objects} JSON objects")
        if self.self_contradictory:
            out.append("objects disagree")
        return out


def find_json_objects(text: str) -> list[dict]:
    """Return every balanced top-level {...} span that parses as an object.

    Scans for brace balance rather than splitting on delimiters, so it works
    whether the repeats are fenced, separated by prose, or run together.
    Strings are tracked so that a brace inside a value does not corrupt the
    depth count.
    """
    if not text:
        return []

    cleaned = _FENCE_LINE.sub(" ", text)
    objects: list[dict] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(cleaned):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(cleaned[start : i + 1])
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1

    return objects


def _end_of_first_object(text: str) -> int:
    """Character index just past the first complete JSON object, or -1."""
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def detect(
    output: str,
    completion_tokens: int = 0,
    max_tokens: int = 0,
) -> Degeneracy:
    """Flag degenerate generation.

    completion_tokens and max_tokens are optional: when absent, only the
    structural checks run. Token-cap detection allows a 1-token margin
    because tokenizers and stop handling occasionally land one short.
    """
    objects = find_json_objects(output)

    hit_cap = bool(max_tokens) and completion_tokens >= max_tokens - 1

    contradictory = False
    if len(objects) > 1:
        first = json.dumps(objects[0], sort_keys=True)
        contradictory = any(json.dumps(o, sort_keys=True) != first for o in objects[1:])

    # Approximate wasted tokens by the share of characters after the first
    # object. Exact accounting would need the tokenizer; the ratio is enough
    # to size the cost.
    wasted = 0
    if completion_tokens and len(objects) > 1:
        end = _end_of_first_object(output)
        if end > 0 and len(output) > 0:
            wasted = int(completion_tokens * (1 - end / len(output)))

    return Degeneracy(
        hit_token_cap=hit_cap,
        n_json_objects=len(objects),
        self_contradictory=contradictory,
        wasted_tokens=max(0, wasted),
    )