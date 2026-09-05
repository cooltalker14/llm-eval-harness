# llm-eval-harness

Regression detection for LLM systems: does this model change actually make things better, or is the difference noise?

This is the companion to [llm-serving-bench](https://github.com/cooltalker14/llm-serving-bench), which found that AWQ int4 quantization delivers **2.2x the throughput of bf16 at low concurrency**. That result is only actionable if the quantized model still produces correct output — a question throughput numbers cannot answer. This repo answers it, and generalizes to any model change: a prompt edit, a version bump, a fine-tune, a new provider.

---

## The problem this solves

The usual model evaluation looks like this:

> We changed the prompt and accuracy went from 74% to 78%. Shipping it.

On a 40-item dataset that difference is **well inside the noise**. The same two configurations re-run with a different random seed could easily reverse it. Teams ship regressions this way routinely, because the eval reported a number without an interval.

This harness reports the interval, the p-value, and — for null results — the smallest difference the dataset was capable of detecting.

```
| Metric                       | baseline | candidate | Delta  | 95% CI           | p     | Verdict         |
|------------------------------|---------:|----------:|-------:|------------------|------:|-----------------|
| Field score (partial credit) |    0.933 |     0.925 | -0.008 | [-0.075, +0.050] | 0.885 | not significant |

With n=40 and a paired standard deviation of 0.207, this dataset can detect a difference of
0.091 at 80% power. Differences smaller than that are not measurable here.

> A non-significant result is not evidence of equivalence.
```

---

## Design decisions

### Comparisons are paired

Both configurations see identical inputs, so per-item scores are correlated. Treating them as independent samples discards that structure and loses most of the statistical power.

The test suite demonstrates the size of the effect: on correlated data with a small consistent shift, the paired analysis produces a confidence interval **42x narrower** than the unpaired equivalent, and detects an effect the unpaired test misses entirely.

Binary outcomes use an exact **McNemar** test, which counts only discordant pairs. Items both configurations get right carry no information about which is better, and including them dilutes the signal.

### Deterministic checks come first

An LLM judge is expensive, slow, and itself a source of variance. Most real regressions — malformed JSON, missing fields, values outside the allowed enum — are catchable with code that costs nothing and never drifts.

The harness separates two things that are usually conflated:

| Failure | Meaning |
|---|---|
| **Contract violation** | Output was unparseable, missing fields, or used a value outside the schema. Never acceptable. |
| **Accuracy failure** | Output was well-formed but wrong. Measured statistically. |

A model that returns `"category": "catastrophe"` has not made a small mistake; it has broken the contract, and the report treats it that way.

### A judge must be calibrated before it is trusted

An uncalibrated LLM judge is a random number generator with good prose. Before any judge score is reported, the harness measures its agreement with human labels using **Cohen's kappa**, not raw agreement.

The difference matters. From the test suite: on a dataset where 90% of items genuinely pass, a judge that blindly answers "pass" every time achieves **90% raw agreement** — and **kappa = 0.000**. The harness marks it *"slight — do not use."*

Kappa thresholds follow Landis & Koch; judges below 0.60 are flagged as unusable rather than silently reported.

### Nothing is trusted until it is verified against known ground truth

Same discipline as the serving benchmark: the measurement machinery is tested against cases where the correct answer is known in advance.

```bash
python tests/test_stats.py      # statistics
python tests/test_scorers.py    # deterministic scoring
python tests/test_agreement.py  # judge calibration
python tests/test_pipeline.py   # end-to-end regression detection
```

What these assert:

- **False positive rate matches alpha.** 200 trials comparing a distribution against itself; significance is claimed in ~5% of them, as it should be.
- **Confidence intervals cover the true effect ~95% of the time.** Measured over 200 trials against a known injected effect.
- **Known effects are recovered.** A +0.10 injected difference is estimated at +0.098.
- **McNemar ignores concordant pairs.** Adding 50 items both configurations get right leaves the p-value bit-identical.
- **Injected regressions are detected end-to-end**, and equivalent runs are not falsely flagged.

All four run in CI on every push. If they fail, no number this repo produces should be believed.

---

## The task

Structured extraction from free-text incident reports. Given a report, produce a JSON record with `category`, `severity`, `component`, and `action_required`.

Chosen because it exercises both scoring paths: three fields have a single defensible answer and are scored by exact match, while `component` is free text where near-synonyms ("auth" vs "authentication") are not errors.

The golden dataset is **40 hand-labelled items** in `data/incidents.jsonl`. Every item carries a `notes` field documenting *why* it was labelled that way:

```json
{
  "id": "inc-022",
  "text": "The staging environment has been down since Friday. Not blocking anyone since everyone is testing locally.",
  "gold": {"category": "outage", "severity": "low", "component": "staging", "action_required": true},
  "notes": "Outage by category, low by severity. Tests that the model does not equate the two."
}
```

Several items are deliberately adversarial: an outage that is genuinely low severity, a CVE whose published rating overstates our actual exposure, a load test posted for the record where the correct answer is that nothing is wrong.

**Forty items is small, and the harness says so.** Every report states the minimum detectable effect for the dataset size rather than letting a null result be misread as equivalence.

---

## Usage

```bash
pip install -r requirements.txt

# generate outputs for each configuration (any OpenAI-compatible endpoint)
python -m evalkit.run --dataset data/incidents.jsonl \
  --base-url http://localhost:8000 --model qwen7b --tag bf16

python -m evalkit.run --dataset data/incidents.jsonl \
  --base-url http://localhost:8000 --model qwen7b-awq --tag awq

# compare
python -m evalkit.compare --baseline runs/bf16.json --candidate runs/awq.json
```

### Gating CI on regressions

```bash
python -m evalkit.compare \
  --baseline runs/production.json \
  --candidate runs/pr.json \
  --fail-on-regression
```

Exits non-zero when any tracked metric degrades significantly, so a pull request that breaks output quality fails the build rather than shipping.

Generations are cached by `(model, prompt, sampling params)`, so re-scoring never triggers re-inference. Scorers change often; model outputs do not.

---

## Layout

```
src/evalkit/dataset.py               schema, validation, prompt
src/evalkit/runner.py                generation with caching and cost tracking
src/evalkit/scorers/deterministic.py JSON, schema, exact-match scoring
src/evalkit/agreement.py             Cohen's kappa, Krippendorff's alpha
src/evalkit/stats.py                 paired bootstrap, McNemar, power analysis
src/evalkit/compare.py               A/B report and CI gate
data/incidents.jsonl                 40 hand-labelled items with rationale
tests/                               four verification suites
```

---

## Limitations

- **Forty items is a starter dataset.** It can detect differences of roughly 0.09 in field score, not smaller ones. Every report states this rather than hiding it.
- **Temperature is fixed at 0.** Comparing two configurations should not also measure sampling noise. Evaluating at non-zero temperature requires multiple samples per item, which is a different experiment.
- **Single labeller.** The dataset carries documented rationale per item, but no second human annotator, so there is no human-human agreement ceiling to compare a judge against. `krippendorff_alpha_nominal` is implemented for when a second labeller exists.
- **The judge scorer is not yet wired to a provider.** Calibration machinery and thresholds are built and tested; connecting a specific judge model is the next step.
- **No drift tracking over time.** Comparisons are pairwise. Tracking quality across many versions is a natural extension.
