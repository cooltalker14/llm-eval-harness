"""Deterministic scorers: cheap, reproducible, and not themselves a model.

Run these before reaching for a judge. They cost nothing, never drift, and
in practice catch most real regressions. A judge is only needed for the
fields where near-synonyms are acceptable.
"""

import json
import re
from dataclasses import dataclass, field

from ..dataset import CATEGORIES, EXACT_FIELDS, SCHEMA_FIELDS, SEVERITIES, Item
from .degeneracy import Degeneracy, detect as detect_degeneracy

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


@dataclass
class ItemScore:
    item_id: str
    parsed: bool                      # output was valid JSON
    schema_ok: bool                   # all required fields present and well-typed
    fields: dict[str, bool] = field(default_factory=dict)   # per-field exact match
    raw_output: str = ""
    parse_error: str = ""
    extracted: dict | None = None
    # Degenerate generation is scored separately from accuracy: a model can
    # give the right answer and then keep talking, which is invisible to
    # every accuracy metric but costs tokens and signals instability.
    degeneracy: Degeneracy = field(default_factory=Degeneracy)

    @property
    def exact_all(self) -> bool:
        """True only if every exactly-scored field matches gold."""
        return self.schema_ok and all(self.fields.get(f, False) for f in EXACT_FIELDS)

    @property
    def field_score(self) -> float:
        """Fraction of exactly-scored fields that match. Partial credit."""
        if not self.fields:
            return 0.0
        return sum(self.fields.get(f, False) for f in EXACT_FIELDS) / len(EXACT_FIELDS)


def extract_json(text: str) -> tuple[dict | None, str]:
    """Parse a JSON object from model output.

    Models wrap JSON in markdown fences or add commentary despite being told
    not to. Stripping that is not cheating: it is a formatting artifact, not
    a semantic error, and conflating the two hides real quality differences.
    Genuinely unparseable output still fails.
    """
    if not text or not text.strip():
        return None, "empty output"

    cleaned = _FENCE.sub("", text).strip()

    try:
        obj = json.loads(cleaned)
        return (obj, "") if isinstance(obj, dict) else (None, "JSON is not an object")
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced {...} span.
    start = cleaned.find("{")
    if start == -1:
        return None, "no JSON object found"
    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    return (obj, "") if isinstance(obj, dict) else (None, "not an object")
                except json.JSONDecodeError as e:
                    return None, f"invalid JSON: {e}"
    return None, "unbalanced braces"


def _normalize(value) -> object:
    """Normalize for comparison. Case and whitespace are not real differences."""
    if isinstance(value, str):
        return value.strip().lower()
    return value


def score_item(
    item: Item,
    output: str,
    completion_tokens: int = 0,
    max_tokens: int = 0,
) -> ItemScore:
    """Score one generation.

    Token counts are optional; supplying them enables token-cap detection,
    which is the clearest signal of a runaway generation.
    """
    degen = detect_degeneracy(output, completion_tokens, max_tokens)

    obj, err = extract_json(output)
    if obj is None:
        return ItemScore(
            item_id=item.id, parsed=False, schema_ok=False,
            raw_output=output, parse_error=err, degeneracy=degen,
        )

    missing = [f for f in SCHEMA_FIELDS if f not in obj]
    schema_ok = not missing

    # Enum membership is part of schema conformance: a category outside the
    # allowed set is a contract violation, not merely a wrong answer.
    if schema_ok:
        if _normalize(obj.get("category")) not in CATEGORIES:
            schema_ok = False
            missing = ["category not in enum"]
        elif _normalize(obj.get("severity")) not in SEVERITIES:
            schema_ok = False
            missing = ["severity not in enum"]
        elif not isinstance(obj.get("action_required"), bool):
            schema_ok = False
            missing = ["action_required not bool"]

    fields = {
        f: _normalize(obj.get(f)) == _normalize(item.gold[f])
        for f in EXACT_FIELDS
    }

    return ItemScore(
        item_id=item.id,
        parsed=True,
        schema_ok=schema_ok,
        fields=fields,
        raw_output=output,
        parse_error="; ".join(missing) if missing else "",
        extracted=obj,
        degeneracy=degen,
    )