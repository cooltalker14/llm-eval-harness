"""Verify judge calibration correctly identifies unusable judges.

The case that matters: a judge that always says "pass" on an imbalanced
dataset scores high raw agreement while carrying no information. Kappa must
catch it. If it does not, the calibration gate is decorative.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalkit.agreement import cohens_kappa, krippendorff_alpha_nominal  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name + " " + detail)


def main():
    print("Verifying judge calibration\n")

    print("Perfect agreement gives kappa 1.0")
    h = [True, False, True, True, False, False, True, False]
    a = cohens_kappa(h, h)
    check("kappa == 1.0", abs(a.kappa - 1.0) < 1e-9, f"({a.summary()})")
    check("marked usable", a.usable)
    print()

    print("A constant judge on imbalanced data is caught by kappa but not raw agreement")
    # 90% of items genuinely pass; judge blindly says pass every time.
    human = [True] * 90 + [False] * 10
    judge = [True] * 100
    a = cohens_kappa(human, judge)
    check("raw agreement is misleadingly high", a.raw_agreement >= 0.85,
          f"({a.raw_agreement:.1%})")
    check("kappa near zero", abs(a.kappa) < 0.05, f"(kappa={a.kappa:.4f})")
    check("marked NOT usable", not a.usable, f"({a.verdict})")
    check("false positives counted", a.false_positives == 10, f"(got {a.false_positives})")
    print()

    print("Random judge scores near zero kappa")
    rng = np.random.default_rng(0)
    human = rng.random(400) < 0.6
    judge = rng.random(400) < 0.6
    a = cohens_kappa(human.tolist(), judge.tolist())
    check("kappa near zero", abs(a.kappa) < 0.15, f"(kappa={a.kappa:.4f})")
    check("marked NOT usable", not a.usable)
    print()

    print("Inverted judge scores negative kappa")
    human = [True, True, False, False, True, False, True, False]
    judge = [not x for x in human]
    a = cohens_kappa(human, judge)
    check("kappa negative", a.kappa < 0, f"(kappa={a.kappa:.3f})")
    check("verdict flags worse than chance", "worse than chance" in a.verdict, f"({a.verdict})")
    print()

    print("A good but imperfect judge is marked usable")
    rng = np.random.default_rng(5)
    human = (rng.random(300) < 0.5)
    judge = human.copy()
    flip = rng.random(300) < 0.08          # 8% disagreement
    judge[flip] = ~judge[flip]
    a = cohens_kappa(human.tolist(), judge.tolist())
    check("kappa substantial", a.kappa >= 0.60, f"(kappa={a.kappa:.3f})")
    check("marked usable", a.usable, f"({a.verdict})")
    print()

    print("Krippendorff alpha handles multiple raters with gaps")
    perfect = [[1, 0, 1, 1, 0], [1, 0, 1, 1, 0]]
    check("perfect multi-rater alpha == 1", abs(krippendorff_alpha_nominal(perfect) - 1.0) < 1e-9,
          f"(alpha={krippendorff_alpha_nominal(perfect):.3f})")

    with_gaps = [[1, 0, 1, 1, None], [1, 0, 1, None, 0]]
    val = krippendorff_alpha_nominal(with_gaps)
    check("tolerates missing labels", not np.isnan(val), f"(alpha={val:.3f})")

    disagree = [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]
    check("total disagreement gives low alpha", krippendorff_alpha_nominal(disagree) < 0,
          f"(alpha={krippendorff_alpha_nominal(disagree):.3f})")
    print()

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS: judge calibration verified")


if __name__ == "__main__":
    main()
