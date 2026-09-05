"""Compare two model configurations on the same golden dataset.

Produces a markdown report with paired statistics, per-field breakdown, and
an explicit regression verdict suitable for gating CI.
"""

import argparse
import json
from pathlib import Path

from .dataset import EXACT_FIELDS, load as load_dataset
from .runner import load as load_run
from .scorers.deterministic import score_item
from .stats import mcnemar, min_detectable_effect, paired_bootstrap


def score_run(items, run):
    """Score every item, passing token counts through for cap detection."""
    by_id = {g.item_id: g for g in run.generations}
    max_tokens = (run.env or {}).get("max_tokens", 0)
    out = []
    for it in items:
        g = by_id.get(it.id)
        out.append(score_item(
            it,
            g.output if g else "",
            completion_tokens=g.completion_tokens if g else 0,
            max_tokens=max_tokens,
        ))
    return out


def build_report(items, run_a, run_b, scores_a, scores_b, alpha=0.05) -> tuple[str, bool]:
    """Return (markdown report, regression_detected)."""
    lines: list[str] = []
    regression = False

    a_tag, b_tag = run_a.tag, run_b.tag
    lines.append(f"# Evaluation: `{b_tag}` vs `{a_tag}`\n")
    lines.append(f"Dataset: {len(items)} items. Baseline `{a_tag}`, candidate `{b_tag}`.\n")

    # --- Contract-level metrics: these are absolute requirements, not scores.
    lines.append("## Output contract\n")
    lines.append("| Metric | " + f"{a_tag} | {b_tag} | Delta |")
    lines.append("|---|---:|---:|---:|")
    for label, attr in [("Parsed as JSON", "parsed"), ("Schema conformant", "schema_ok")]:
        pa = sum(getattr(s, attr) for s in scores_a) / len(scores_a)
        pb = sum(getattr(s, attr) for s in scores_b) / len(scores_b)
        lines.append(f"| {label} | {pa:.1%} | {pb:.1%} | {pb-pa:+.1%} |")
        if pb < pa:
            regression = True
    lines.append("")

    # --- Accuracy with paired statistics.
    lines.append("## Accuracy\n")

    partial_a = [s.field_score for s in scores_a]
    partial_b = [s.field_score for s in scores_b]
    boot = paired_bootstrap(partial_a, partial_b)

    exact_a = [s.exact_all for s in scores_a]
    exact_b = [s.exact_all for s in scores_b]
    mc = mcnemar(exact_a, exact_b)

    lines.append("| Metric | " + f"{a_tag} | {b_tag} | Delta | 95% CI | p | Verdict |")
    lines.append("|---|---:|---:|---:|---|---:|---|")
    lines.append(
        f"| Field score (partial credit) | {boot.mean_a:.3f} | {boot.mean_b:.3f} "
        f"| {boot.delta:+.3f} | [{boot.ci_low:+.3f}, {boot.ci_high:+.3f}] "
        f"| {boot.p_value:.3f} | {'**significant**' if boot.significant else 'not significant'} |"
    )
    lines.append(
        f"| All fields correct | {mc.mean_a:.3f} | {mc.mean_b:.3f} "
        f"| {mc.delta:+.3f} | [{mc.ci_low:+.3f}, {mc.ci_high:+.3f}] "
        f"| {mc.p_value:.3f} | {'**significant**' if mc.significant else 'not significant'} |"
    )
    lines.append("")
    lines.append(f"Bootstrap: {boot.method}. Binary: {mc.method}.\n")

    if boot.significant and boot.delta < 0:
        regression = True
    if mc.significant and mc.delta < 0:
        regression = True

    # --- The number that makes a null result interpretable.
    import numpy as np
    sd = float(np.std(np.array(partial_b) - np.array(partial_a), ddof=1))
    mde = min_detectable_effect(len(items), sd)
    lines.append("### Statistical power\n")
    if sd == 0:
        lines.append("The two configurations produced identical field scores on every item.\n")
    else:
        lines.append(
            f"With n={len(items)} and a paired standard deviation of {sd:.3f}, this dataset "
            f"can detect a difference of **{mde:.3f}** at 80% power. Differences smaller than "
            f"that are not measurable here regardless of what the point estimate shows.\n"
        )
        if not boot.significant:
            lines.append(
                "> A non-significant result is **not** evidence of equivalence. It means any "
                f"difference is smaller than {mde:.3f}, or the dataset is too small to see it.\n"
            )

    # --- Per-field breakdown localises the regression.
    lines.append("## Per-field accuracy\n")
    lines.append("| Field | " + f"{a_tag} | {b_tag} | Delta | p |")
    lines.append("|---|---:|---:|---:|---:|")
    for fname in EXACT_FIELDS:
        fa = [s.fields.get(fname, False) for s in scores_a]
        fb = [s.fields.get(fname, False) for s in scores_b]
        r = mcnemar(fa, fb)
        flag = " **regression**" if r.significant and r.delta < 0 else ""
        lines.append(
            f"| {fname} | {r.mean_a:.1%} | {r.mean_b:.1%} | {r.delta:+.1%} | {r.p_value:.3f}{flag} |"
        )
    lines.append("")

    # --- Degenerate generation: invisible to accuracy, expensive in production.
    lines.append("## Generation health\n")
    lines.append(
        "A model can answer correctly and then keep generating: repeating the object, "
        "reasoning aloud, or looping to the token cap. The extractor recovers the first "
        "object, so accuracy metrics miss this entirely.\n"
    )

    da = [s.degeneracy for s in scores_a]
    db = [s.degeneracy for s in scores_b]
    n = len(items)

    lines.append("| Signal | " + f"{a_tag} | {b_tag} |")
    lines.append("|---|---:|---:|")
    for label, fn in [
        ("Degenerate generations", lambda d: d.is_degenerate),
        ("Hit token cap", lambda d: d.hit_token_cap),
        ("Emitted >1 JSON object", lambda d: d.multiple_objects),
        ("Objects contradict each other", lambda d: d.self_contradictory),
    ]:
        ca, cb = sum(fn(d) for d in da), sum(fn(d) for d in db)
        lines.append(f"| {label} | {ca} ({ca/n:.1%}) | {cb} ({cb/n:.1%}) |")

    wa, wb = sum(d.wasted_tokens for d in da), sum(d.wasted_tokens for d in db)
    lines.append(f"| Tokens generated after first object | ~{wa} | ~{wb} |")
    lines.append("")

    # Any token-cap hit is worth naming regardless of rate: it is the most
    # severe form, and at these counts no test could call it significant.
    for tag, scores in [(a_tag, scores_a), (b_tag, scores_b)]:
        bad = [s for s in scores if s.degeneracy.is_degenerate]
        if bad:
            lines.append(f"**{tag}**: " + "; ".join(
                f"`{s.item_id}` ({', '.join(s.degeneracy.reasons)})" for s in bad[:6]
            ))
    lines.append("")

    cap_a = sum(d.hit_token_cap for d in da)
    cap_b = sum(d.hit_token_cap for d in db)
    if cap_b > cap_a:
        regression = True
        lines.append(
            f"> Candidate hit the token cap {cap_b - cap_a} more time(s) than baseline. "
            "Treated as a regression: runaway generation is a production failure "
            "even when the extracted answer is correct.\n"
        )
    elif sum(d.is_degenerate for d in db) > sum(d.is_degenerate for d in da):
        lines.append(
            "> Candidate shows more degenerate generations than baseline. Not gated "
            "(counts this small are not statistically testable), but worth watching.\n"
        )

    # --- Cost and latency: quality is not the only axis that matters.
    lines.append("## Cost and latency\n")
    lines.append("| Metric | " + f"{a_tag} | {b_tag} |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Completion tokens | {run_a.total_completion_tokens:,} | {run_b.total_completion_tokens:,} |")
    lines.append(f"| Wall time (s) | {run_a.wall_time_s:.1f} | {run_b.wall_time_s:.1f} |")
    lines.append(f"| Errors | {run_a.n_errors} | {run_b.n_errors} |")
    lines.append("")

    # --- Named failures, so the report is actionable rather than just a verdict.
    newly_broken = [
        (items[i].id, items[i].text[:70])
        for i in range(len(items))
        if exact_a[i] and not exact_b[i]
    ]
    if newly_broken:
        lines.append(f"## Items `{b_tag}` got wrong that `{a_tag}` got right ({len(newly_broken)})\n")
        for iid, text in newly_broken[:15]:
            lines.append(f"- `{iid}` {text}...")
        if len(newly_broken) > 15:
            lines.append(f"- ...and {len(newly_broken)-15} more")
        lines.append("")

    lines.append("## Verdict\n")
    lines.append(
        f"**{'REGRESSION DETECTED' if regression else 'NO REGRESSION'}** — "
        + ("candidate is significantly worse on at least one tracked metric."
           if regression else
           "no tracked metric degraded significantly.")
    )

    return "\n".join(lines), regression


def main():
    ap = argparse.ArgumentParser(description="Compare two eval runs")
    ap.add_argument("--dataset", type=Path, default=Path("data/incidents.jsonl"))
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/comparison.md"))
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit 1 if a significant regression is found (for CI)")
    args = ap.parse_args()

    items = load_dataset(args.dataset)
    run_a, run_b = load_run(args.baseline), load_run(args.candidate)

    scores_a = score_run(items, run_a)
    scores_b = score_run(items, run_b)

    report, regression = build_report(items, run_a, run_b, scores_a, scores_b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"\nwrote {args.out}")

    if regression and args.fail_on_regression:
        raise SystemExit(1)


if __name__ == "__main__":
    main()