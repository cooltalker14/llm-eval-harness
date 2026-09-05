"""Statistics for comparing two model configurations on the same dataset.

The central design choice here: comparisons are PAIRED. Both configurations
see identical inputs, so per-item scores are correlated and treating them as
independent samples throws away that structure. A paired test on the per-item
differences is both more powerful and more correct.

Getting this wrong is the most common failure in model evaluation: reporting
"78% vs 74%" with no interval, when the difference is well inside the noise
for a dataset that size.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    delta: float          # mean_b - mean_a
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    effect_size: float    # Cohen's d for paired samples
    method: str

    def summary(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (
            f"delta={self.delta:+.4f} "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}] "
            f"p={self.p_value:.4f} ({verdict}, n={self.n})"
        )


def paired_bootstrap(
    a: list[float],
    b: list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    """Bootstrap CI on the mean paired difference (b - a).

    Resamples ITEMS, not scores independently, which preserves the pairing.
    Works for any score in [0,1] including continuous judge scores, and makes
    no normality assumption.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired arrays must match: {arr_a.shape} vs {arr_b.shape}")
    if arr_a.size == 0:
        raise ValueError("cannot compare empty arrays")

    diffs = arr_b - arr_a
    n = diffs.size
    observed = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = diffs[idx].mean(axis=1)

    ci_low = float(np.percentile(boot, 100 * alpha / 2))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))

    # Two-sided p-value by inverting the CI: how much of the bootstrap
    # distribution falls on the opposite side of zero from the observed effect.
    if observed >= 0:
        p = 2 * float((boot <= 0).mean())
    else:
        p = 2 * float((boot >= 0).mean())
    p = min(1.0, p)

    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    d = observed / sd if sd > 0 else 0.0

    return PairedResult(
        n=n,
        mean_a=float(arr_a.mean()),
        mean_b=float(arr_b.mean()),
        delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        significant=not (ci_low <= 0 <= ci_high),
        effect_size=d,
        method="paired bootstrap",
    )


def mcnemar(a: list[bool], b: list[bool]) -> PairedResult:
    """Exact McNemar test for paired binary outcomes (pass/fail per item).

    Only the discordant pairs carry information: items where one config passed
    and the other failed. Items both got right or both got wrong tell you
    nothing about which is better, and including them inflates apparent
    agreement.
    """
    arr_a = np.asarray(a, dtype=bool)
    arr_b = np.asarray(b, dtype=bool)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired arrays must match: {arr_a.shape} vs {arr_b.shape}")

    n = arr_a.size
    b_only = int((~arr_a & arr_b).sum())   # b passed, a failed
    a_only = int((arr_a & ~arr_b).sum())   # a passed, b failed
    n_disc = a_only + b_only

    if n_disc == 0:
        p = 1.0
    else:
        # Exact binomial: under H0 each discordant pair is a coin flip.
        p = float(sps.binomtest(b_only, n_disc, 0.5).pvalue)

    delta = float(arr_b.mean() - arr_a.mean())

    # Wilson interval on the discordant proportion, mapped back to the
    # accuracy-difference scale. Avoids the normal approximation failing
    # when discordant counts are small.
    if n_disc > 0:
        lo, hi = _wilson(b_only, n_disc)
        ci_low = (2 * lo - 1) * n_disc / n
        ci_high = (2 * hi - 1) * n_disc / n
    else:
        ci_low = ci_high = 0.0

    return PairedResult(
        n=n,
        mean_a=float(arr_a.mean()),
        mean_b=float(arr_b.mean()),
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        significant=p < 0.05,
        effect_size=(b_only - a_only) / n_disc if n_disc else 0.0,
        method=f"McNemar exact (discordant: {b_only} b-only, {a_only} a-only)",
    )


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def min_detectable_effect(n: int, sd: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest paired difference detectable at the given power.

    Report this alongside any null result. "No significant difference" is
    meaningless without saying what difference the dataset was large enough
    to find; a 40-item set cannot detect a 2-point change and claiming
    equivalence from one is a mistake.
    """
    if n < 2 or sd <= 0:
        return float("nan")
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(power)
    return float((z_a + z_b) * sd / np.sqrt(n))
