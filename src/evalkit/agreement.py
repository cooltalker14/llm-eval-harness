"""Measure whether an LLM judge agrees with human labels well enough to use.

An uncalibrated judge is a random number generator with good prose. Before
any judge score is reported, it must be checked against a human-labelled
subset — otherwise you are measuring the judge, not the model.

Raw agreement is not sufficient. On a task where 90% of items are "good",
a judge that always says "good" scores 90% agreement while carrying zero
information. Cohen's kappa corrects for agreement expected by chance.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Agreement:
    n: int
    raw_agreement: float
    kappa: float
    judge_positive_rate: float
    human_positive_rate: float
    false_positives: int      # judge said pass, human said fail
    false_negatives: int      # judge said fail, human said pass

    @property
    def verdict(self) -> str:
        """Landis & Koch benchmarks, widely used for kappa interpretation."""
        k = self.kappa
        if k < 0.0:
            return "worse than chance - do not use"
        if k < 0.20:
            return "slight - do not use"
        if k < 0.40:
            return "fair - unreliable, treat scores as directional only"
        if k < 0.60:
            return "moderate - usable with caution"
        if k < 0.80:
            return "substantial - usable"
        return "almost perfect - usable"

    @property
    def usable(self) -> bool:
        return self.kappa >= 0.60

    def summary(self) -> str:
        return (
            f"kappa={self.kappa:.3f} ({self.verdict}), "
            f"raw agreement={self.raw_agreement:.1%}, n={self.n}, "
            f"FP={self.false_positives} FN={self.false_negatives}"
        )


def cohens_kappa(human: list[bool], judge: list[bool]) -> Agreement:
    """Cohen's kappa between a human labeller and an LLM judge."""
    h = np.asarray(human, dtype=bool)
    j = np.asarray(judge, dtype=bool)
    if h.shape != j.shape:
        raise ValueError(f"label arrays must match: {h.shape} vs {j.shape}")
    if h.size == 0:
        raise ValueError("cannot compute agreement on empty labels")

    n = h.size
    observed = float((h == j).mean())

    # Chance agreement from the marginal rates of each rater.
    ph, pj = float(h.mean()), float(j.mean())
    expected = ph * pj + (1 - ph) * (1 - pj)

    # Perfect agreement with degenerate marginals leaves kappa undefined;
    # report 1.0 when both raters agree on everything, 0.0 otherwise.
    if expected >= 1.0:
        kappa = 1.0 if observed >= 1.0 else 0.0
    else:
        kappa = (observed - expected) / (1 - expected)

    return Agreement(
        n=n,
        raw_agreement=observed,
        kappa=float(kappa),
        judge_positive_rate=pj,
        human_positive_rate=ph,
        false_positives=int((~h & j).sum()),
        false_negatives=int((h & ~j).sum()),
    )


def krippendorff_alpha_nominal(ratings: list[list[int | None]]) -> float:
    """Krippendorff's alpha for 2+ raters on nominal data, allowing gaps.

    Used when more than one human labels the calibration subset, to establish
    the human-human ceiling. A judge cannot meaningfully be expected to beat
    the agreement humans reach with each other.
    """
    arr = [[v for v in col] for col in zip(*ratings)]
    pairable = [col for col in arr if sum(v is not None for v in col) >= 2]
    if not pairable:
        return float("nan")

    values: list[int] = [v for col in pairable for v in col if v is not None]
    categories = sorted(set(values))
    if len(categories) < 2:
        return 1.0

    observed_disagreement = 0.0
    total_pairs = 0
    for col in pairable:
        present = [v for v in col if v is not None]
        m = len(present)
        for i in range(m):
            for k in range(m):
                if i != k:
                    observed_disagreement += present[i] != present[k]
                    total_pairs += 1
    Do = observed_disagreement / total_pairs if total_pairs else 0.0

    counts = {c: values.count(c) for c in categories}
    total = len(values)
    De = 1.0 - sum((cnt / total) ** 2 for cnt in counts.values())
    De *= total / (total - 1) if total > 1 else 1.0

    return 1.0 - Do / De if De > 0 else 1.0
