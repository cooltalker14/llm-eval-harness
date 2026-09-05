"""Verify degenerate-output detection on cases with known correct answers.

The property that matters: a generation can be fully correct by accuracy
metrics while still misbehaving. These flags must fire on that case and stay
quiet on clean output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) # noqa: E402
from evalkit.dataset import Item  # noqa: E402
from evalkit.scorers.degeneracy import detect, find_json_objects  # noqa: E402
from evalkit.scorers.deterministic import score_item  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name + " " + detail)


OBJ = '{"category": "outage", "severity": "critical", "component": "checkout", "action_required": true}'
ITEM = Item("t1", "Checkout is down.",
            {"category": "outage", "severity": "critical",
             "component": "checkout", "action_required": True})


def main():
    print("Verifying degenerate-output detection\n")

    print("Object counting handles the formats models actually produce")
    cases = [
        ("single object", OBJ, 1),
        ("two run together", f"{OBJ} {OBJ}", 2),
        ("separated by prose", f"{OBJ}\nJSON:\n{OBJ}", 2),
        ("fenced repeat", f"{OBJ}\n```json\n{OBJ}\n```", 2),
        ("four repeats", " ".join([OBJ] * 4), 4),
        ("prose only", "This is an outage.", 0),
        ("nested braces counted once", '{"a": {"b": 1}, "c": 2}', 1),
        ("brace inside a string value", '{"component": "weird}name", "x": 1}', 1),
    ]
    for name, text, expected in cases:
        got = len(find_json_objects(text))
        check(f"count: {name}", got == expected, f"(expected {expected}, got {got})")
    print()

    print("Clean output raises no flags")
    d = detect(OBJ, completion_tokens=33, max_tokens=200)
    check("not degenerate", not d.is_degenerate)
    check("no cap hit", not d.hit_token_cap)
    check("no wasted tokens", d.wasted_tokens == 0, f"(got {d.wasted_tokens})")
    print()

    print("Token cap is detected, with a one-token margin")
    check("at cap", detect(OBJ, 200, 200).hit_token_cap)
    check("one below cap", detect(OBJ, 199, 200).hit_token_cap)
    check("well below cap", not detect(OBJ, 150, 200).hit_token_cap)
    check("no cap info means no cap flag", not detect(OBJ, 200, 0).hit_token_cap)
    print()

    print("Identical repeats are flagged but not called contradictory")
    d = detect(f"{OBJ} {OBJ}", 66, 200)
    check("multiple objects", d.multiple_objects)
    check("degenerate", d.is_degenerate)
    check("not contradictory", not d.self_contradictory)
    check("wasted tokens counted", d.wasted_tokens > 0, f"(~{d.wasted_tokens})")
    print()

    print("Disagreeing repeats are flagged as contradictory")
    other = OBJ.replace("critical", "low")
    d = detect(f"{OBJ} {other}", 66, 200)
    check("contradictory", d.self_contradictory)
    check("reasons list is populated", len(d.reasons) >= 2, f"({d.reasons})")
    print()

    print("A correct answer can still be a degenerate generation")
    s = score_item(ITEM, f"{OBJ}\nJSON:\n{OBJ.replace('critical', 'low')}",
                   completion_tokens=200, max_tokens=200)
    check("accuracy unaffected", s.exact_all, "(first object is correct)")
    check("schema still valid", s.schema_ok)
    check("but flagged degenerate", s.degeneracy.is_degenerate, f"({s.degeneracy.reasons})")
    print("        ^ this is the case the whole check exists for")
    print()

    print("Unparseable output does not crash the detector")
    d = detect("I cannot answer that.", 12, 200)
    check("no crash", True)
    check("zero objects", d.n_json_objects == 0)
    check("not degenerate", not d.is_degenerate)

    d = detect('{"truncated": ', 200, 200)
    check("truncated: cap still flagged", d.hit_token_cap)
    check("truncated: no complete object", d.n_json_objects == 0)
    print()

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS: degenerate-output detection verified")


if __name__ == "__main__":
    main()