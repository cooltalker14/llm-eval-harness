"""Verify the deterministic scorers on outputs with known correct scores.

Covers the formats models actually produce: clean JSON, fenced JSON, JSON
with commentary, truncated output, wrong enum values, and missing fields.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalkit.dataset import Item  # noqa: E402
from evalkit.scorers.deterministic import extract_json, score_item  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name + " " + detail)


ITEM = Item(
    id="t1",
    text="Checkout is down for everyone.",
    gold={"category": "outage", "severity": "critical",
          "component": "checkout", "action_required": True},
)

GOOD = '{"category":"outage","severity":"critical","component":"checkout","action_required":true}'


def main():
    print("Verifying deterministic scorers\n")

    print("JSON extraction handles real-world output formats")
    cases = [
        ("clean", GOOD, True),
        ("markdown fenced", f"```json\n{GOOD}\n```", True),
        ("bare fence", f"```\n{GOOD}\n```", True),
        ("leading commentary", f"Here is the record:\n{GOOD}", True),
        ("trailing commentary", f"{GOOD}\nLet me know if you need more.", True),
        ("empty", "", False),
        ("prose only", "This looks like an outage to me.", False),
        ("truncated", '{"category":"outage","severity":', False),
        ("json array", '["outage","critical"]', False),
    ]
    for name, text, should_parse in cases:
        obj, err = extract_json(text)
        check(f"parse: {name}", (obj is not None) == should_parse, f"({err})" if err else "")
    print()

    print("Correct output scores as fully correct")
    s = score_item(ITEM, GOOD)
    check("parsed", s.parsed)
    check("schema ok", s.schema_ok)
    check("all fields match", s.exact_all)
    check("field score is 1.0", s.field_score == 1.0, f"(got {s.field_score})")
    print()

    print("Case and whitespace differences are not treated as errors")
    s = score_item(ITEM, '{"category":" OUTAGE ","severity":"Critical","component":"Checkout","action_required":true}')
    check("normalizes case and whitespace", s.exact_all, f"(fields={s.fields})")
    print()

    print("Wrong values are caught")
    s = score_item(ITEM, '{"category":"bug","severity":"low","component":"checkout","action_required":true}')
    check("schema still valid", s.schema_ok)
    check("not all correct", not s.exact_all)
    check("partial credit correct", abs(s.field_score - 1/3) < 1e-9, f"(got {s.field_score:.3f})")
    print()

    print("Contract violations fail schema, not just accuracy")
    s = score_item(ITEM, '{"category":"catastrophe","severity":"critical","component":"checkout","action_required":true}')
    check("enum violation fails schema", not s.schema_ok, f"({s.parse_error})")

    s = score_item(ITEM, '{"category":"outage","severity":"critical","component":"checkout","action_required":"yes"}')
    check("wrong type fails schema", not s.schema_ok, f"({s.parse_error})")

    s = score_item(ITEM, '{"category":"outage","severity":"critical"}')
    check("missing fields fail schema", not s.schema_ok, f"({s.parse_error})")
    print()

    print("Unparseable output scores zero rather than crashing")
    s = score_item(ITEM, "I cannot help with that.")
    check("no crash", True)
    check("parsed false", not s.parsed)
    check("field score zero", s.field_score == 0.0)
    check("exact_all false", not s.exact_all)
    print()

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS: deterministic scorers verified")


if __name__ == "__main__":
    main()
