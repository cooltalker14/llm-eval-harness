"""End-to-end test: inject a known regression, confirm the pipeline finds it.

Verifies the property that matters for CI gating — that the harness catches
real degradation and does not cry wolf on equivalent runs. Uses synthetic
generations so it needs no GPU and runs in CI.
"""

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalkit.compare import build_report, score_run  # noqa: E402
from evalkit.dataset import CATEGORIES, SEVERITIES, load as load_dataset  # noqa: E402
from evalkit.runner import Generation, RunResult  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "incidents.jsonl"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name + " " + detail)


def synth(items, tag: str, accuracy: float, seed: int, force_wrong: set[str] = frozenset()):
    """Build a fake run at a target accuracy, optionally breaking specific items."""
    rng = random.Random(seed)
    gens = []
    for it in items:
        g = dict(it.gold)
        if it.id in force_wrong or rng.random() > accuracy:
            g["category"] = rng.choice([c for c in CATEGORIES if c != g["category"]])
        if rng.random() > accuracy:
            g["severity"] = rng.choice([s for s in SEVERITIES if s != g["severity"]])
        gens.append(Generation(it.id, json.dumps(g), 180, 40, 0.3))
    return RunResult(tag=tag, model="synthetic", generations=gens,
                     total_prompt_tokens=180 * len(items),
                     total_completion_tokens=40 * len(items), wall_time_s=12.0)


def compare(items, a, b):
    return build_report(items, a, b, score_run(items, a), score_run(items, b))


def main():
    print("End-to-end pipeline verification\n")
    items = load_dataset(DATA)
    check("dataset loads", len(items) == 40, f"({len(items)} items)")
    print()

    print("Equivalent runs must NOT be flagged as a regression")
    a = synth(items, "baseline", 0.92, seed=1)
    b = synth(items, "candidate", 0.92, seed=2)
    _, reg = compare(items, a, b)
    check("no false regression", not reg)
    print()

    print("A large injected regression MUST be detected")
    broken = {it.id for it in items[:12]}
    c = synth(items, "candidate", 0.92, seed=3, force_wrong=broken)
    report, reg = compare(items, a, c)
    check("regression detected", reg)
    check("verdict in report", "REGRESSION DETECTED" in report)
    check("failing items named", "inc-001" in report)
    print()

    print("Broken output contract is caught even when parsing fails")
    d = RunResult(tag="broken", model="synthetic",
                  generations=[Generation(it.id, "I cannot answer that.") for it in items])
    report, reg = compare(items, a, d)
    check("unparseable output flagged", reg)
    check("contract table shows 0%", "0.0%" in report)
    print()

    print("Identical runs report zero delta")
    report, reg = compare(items, a, a)
    check("no regression against self", not reg)
    check("delta is zero", "+0.000" in report or "identical" in report)
    print()

    print("Report is written to disk and is non-trivial")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.md"
        rep, _ = compare(items, a, c)
        p.write_text(rep)
        check("report written", p.exists() and len(p.read_text()) > 800,
              f"({len(p.read_text())} chars)")
    print()

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS: pipeline detects injected regressions and avoids false alarms")


if __name__ == "__main__":
    main()
