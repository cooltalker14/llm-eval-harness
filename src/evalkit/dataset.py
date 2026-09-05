"""Golden dataset loading, schema definition, and the extraction prompt.

The task: given free-text incident report, extract a structured record.
Chosen because it has both objectively checkable fields (categorical labels
with a single defensible answer) and a fuzzy one, which exercises both the
deterministic and judge scoring paths.
"""

import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = [
    "outage", "bug", "regression", "performance", "capacity",
    "security", "documentation", "question", "feature_request",
]
SEVERITIES = ["critical", "high", "medium", "low"]

SCHEMA_FIELDS = ["category", "severity", "component", "action_required"]

# Fields where the gold label is a single defensible answer, scored by exact
# match. `component` is excluded: it is free text and near-synonyms
# ("auth" vs "authentication") are not errors, so it goes to the judge.
EXACT_FIELDS = ["category", "severity", "action_required"]

PROMPT = """You are triaging an incident report. Extract a structured record.

Respond with ONLY a JSON object, no markdown fences and no commentary, with exactly these keys:
  "category": one of {categories}
  "severity": one of {severities}
  "component": short lowercase name of the affected system (1-2 words)
  "action_required": true if someone needs to take action, false otherwise

Incident report:
{text}

JSON:"""


@dataclass
class Item:
    id: str
    text: str
    gold: dict
    notes: str = ""

    def prompt(self) -> str:
        return PROMPT.format(
            categories=", ".join(CATEGORIES),
            severities=", ".join(SEVERITIES),
            text=self.text,
        )


def load(path: str | Path) -> list[Item]:
    """Load and validate the golden dataset.

    Validation is strict and fails loudly: a silently malformed gold label
    corrupts every downstream comparison, and that failure is very hard to
    notice once it is buried in an aggregate score.
    """
    items: list[Item] = []
    seen: set[str] = set()

    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno} invalid JSON: {e}") from e

        for key in ("id", "text", "gold"):
            if key not in raw:
                raise ValueError(f"{path}:{lineno} missing required key '{key}'")

        if raw["id"] in seen:
            raise ValueError(f"{path}:{lineno} duplicate id '{raw['id']}'")
        seen.add(raw["id"])

        gold = raw["gold"]
        for field in SCHEMA_FIELDS:
            if field not in gold:
                raise ValueError(f"{path}:{lineno} gold missing field '{field}'")
        if gold["category"] not in CATEGORIES:
            raise ValueError(f"{path}:{lineno} bad category '{gold['category']}'")
        if gold["severity"] not in SEVERITIES:
            raise ValueError(f"{path}:{lineno} bad severity '{gold['severity']}'")
        if not isinstance(gold["action_required"], bool):
            raise ValueError(f"{path}:{lineno} action_required must be bool")

        items.append(
            Item(id=raw["id"], text=raw["text"], gold=gold, notes=raw.get("notes", ""))
        )

    if not items:
        raise ValueError(f"{path} contains no items")
    return items
