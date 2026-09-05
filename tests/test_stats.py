"""Verify the comparison statistics against data with known ground truth.

Same discipline as the timing-harness test in llm-serving-bench: before
trusting any real result, confirm the measurement machinery gives the right
answer on cases where the answer is known in advance.

Four properties are checked:
  1. No false positives on identical distributions (calibration)
  2. Real differences are detected (power)
  3. Bootstrap CIs cover the true effect at the stated rate
  4. McNemar only counts discordant pairs
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalkit.stats import (  # noqa: E402
    mcnemar,
    min_detectable_effect,
    paired_bootstrap,
)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(f"{name} {detail}")


def test_no_false_positives():
    """Two samples from the same distribution should rarely be called different.

    Runs 200 trials; at alpha=0.05 we expect ~5% false positives and allow
    up to 10% before flagging the test as miscalibrated.
    """
    rng = np.random.default_rng(42)
    hits = 0
    trials = 200
    for i in range(trials):
        a = rng.normal(0.75, 0.15, 100)
        b = rng.normal(0.75, 0.15, 100)
        if paired_bootstrap(a.tolist(), b.tolist(), n_resamples=2000, seed=i).significant:
            hits += 1
    rate = hits / trials
    check("false positive rate near alpha", 0.0 <= rate <= 0.10, f"(got {rate:.1%}, expect ~5%)")


def test_detects_real_difference():
    """A large, consistent improvement must be detected."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.60, 0.10, 200)
    b = a + rng.normal(0.10, 0.02, 200)   # b is reliably ~0.10 better
    r = paired_bootstrap(a.tolist(), b.tolist(), seed=1)
    check("detects +0.10 effect", r.significant, f"({r.summary()})")
    check("recovers effect size", abs(r.delta - 0.10) < 0.02, f"(delta={r.delta:.4f})")


def test_ci_coverage():
    """95% CIs should contain the true effect about 95% of the time."""
    rng = np.random.default_rng(3)
    true_effect = 0.05
    covered = 0
    trials = 200
    for i in range(trials):
        a = rng.normal(0.70, 0.12, 80)
        b = a + rng.normal(true_effect, 0.05, 80)
        r = paired_bootstrap(a.tolist(), b.tolist(), n_resamples=2000, seed=i)
        if r.ci_low <= true_effect <= r.ci_high:
            covered += 1
    rate = covered / trials
    check("CI coverage near 95%", 0.88 <= rate <= 1.0, f"(got {rate:.1%})")


def test_pairing_matters():
    """Paired analysis must beat unpaired when scores are correlated.

    Constructs items with large per-item variance but a small consistent
    shift. Treating the samples as independent buries the effect in the
    between-item spread; pairing recovers it.
    """
    rng = np.random.default_rng(11)
    difficulty = rng.normal(0.5, 0.30, 150)      # large item-to-item spread
    a = np.clip(difficulty, 0, 1)
    b = np.clip(difficulty + 0.04, 0, 1)         # small consistent gain

    paired = paired_bootstrap(a.tolist(), b.tolist(), seed=5)

    # Unpaired equivalent: shuffle b so the pairing is destroyed.
    shuffled = b.copy()
    rng.shuffle(shuffled)
    unpaired = paired_bootstrap(a.tolist(), shuffled.tolist(), seed=5)

    check("paired detects small consistent effect", paired.significant, f"({paired.summary()})")
    check(
        "unpaired CI is wider (pairing adds power)",
        (unpaired.ci_high - unpaired.ci_low) > (paired.ci_high - paired.ci_low),
        f"(paired width={paired.ci_high-paired.ci_low:.4f}, "
        f"unpaired width={unpaired.ci_high-unpaired.ci_low:.4f})",
    )


def test_mcnemar_ignores_concordant():
    """Adding items both configs get right must not change the p-value.

    This is the defining property of McNemar: only disagreements carry
    information about which config is better.
    """
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 6 + [False] * 4   # b wins 6 discordant, loses 0

    r1 = mcnemar(a, b)
    # Append 50 items both get right.
    r2 = mcnemar(a + [True] * 50, b + [True] * 50)

    check("McNemar p unchanged by concordant pairs", abs(r1.p_value - r2.p_value) < 1e-12,
          f"(p1={r1.p_value:.6f}, p2={r2.p_value:.6f})")
    check("McNemar detects one-sided wins", r1.significant, f"(p={r1.p_value:.4f})")


def test_mcnemar_no_difference():
    """Symmetric disagreement means no evidence either way."""
    a = [True] * 10 + [False] * 10
    b = [False] * 5 + [True] * 5 + [True] * 5 + [False] * 5
    r = mcnemar(a, b)
    check("McNemar null on symmetric disagreement", not r.significant, f"(p={r.p_value:.4f})")


def test_mde_sanity():
    """Minimum detectable effect must shrink as the dataset grows."""
    small = min_detectable_effect(n=40, sd=0.3)
    large = min_detectable_effect(n=400, sd=0.3)
    check("MDE shrinks with n", large < small, f"(n=40: {small:.3f}, n=400: {large:.3f})")
    check("MDE roughly follows 1/sqrt(n)", abs(small / large - np.sqrt(10)) < 0.5,
          f"(ratio={small/large:.2f}, expect {np.sqrt(10):.2f})")


def main():
    print("Verifying comparison statistics against known ground truth\n")
    for fn in [
        test_no_false_positives,
        test_detects_real_difference,
        test_ci_coverage,
        test_pairing_matters,
        test_mcnemar_ignores_concordant,
        test_mcnemar_no_difference,
        test_mde_sanity,
    ]:
        print(f"{fn.__doc__.splitlines()[0]}")
        fn()
        print()

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS: statistics verified against known ground truth")


if __name__ == "__main__":
    main()
