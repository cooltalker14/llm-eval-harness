# Evaluation: `awq` vs `bf16`

Dataset: 40 items. Baseline `bf16`, candidate `awq`.

## Output contract

| Metric | bf16 | awq | Delta |
|---|---:|---:|---:|
| Parsed as JSON | 100.0% | 100.0% | +0.0% |
| Schema conformant | 100.0% | 100.0% | +0.0% |

## Accuracy

| Metric | bf16 | awq | Delta | 95% CI | p | Verdict |
|---|---:|---:|---:|---|---:|---|
| Field score (partial credit) | 0.683 | 0.700 | +0.017 | [-0.025, +0.067] | 0.438 | not significant |
| All fields correct | 0.225 | 0.300 | +0.075 | [-0.031, +0.116] | 0.375 | not significant |

Bootstrap: paired bootstrap. Binary: McNemar exact (discordant: 4 b-only, 1 a-only).

### Statistical power

With n=40 and a paired standard deviation of 0.150, this dataset can detect a difference of **0.066** at 80% power. Differences smaller than that are not measurable here regardless of what the point estimate shows.

> A non-significant result is **not** evidence of equivalence. It means any difference is smaller than 0.066, or the dataset is too small to see it.

## Per-field accuracy

| Field | bf16 | awq | Delta | p |
|---|---:|---:|---:|---:|
| category | 62.5% | 70.0% | +7.5% | 0.250 |
| severity | 55.0% | 52.5% | -2.5% | 1.000 |
| action_required | 87.5% | 87.5% | +0.0% | 1.000 |

## Generation health

A model can answer correctly and then keep generating: repeating the object, reasoning aloud, or looping to the token cap. The extractor recovers the first object, so accuracy metrics miss this entirely.

| Signal | bf16 | awq |
|---|---:|---:|
| Degenerate generations | 2 (5.0%) | 2 (5.0%) |
| Hit token cap | 0 (0.0%) | 1 (2.5%) |
| Emitted >1 JSON object | 2 (5.0%) | 2 (5.0%) |
| Objects contradict each other | 0 (0.0%) | 1 (2.5%) |
| Tokens generated after first object | ~84 | ~202 |

**bf16**: `inc-007` (2 JSON objects); `inc-013` (2 JSON objects)
**awq**: `inc-007` (2 JSON objects); `inc-022` (hit token cap, 4 JSON objects, objects disagree)

> Candidate hit the token cap 1 more time(s) than baseline. Treated as a regression: runaway generation is a production failure even when the extracted answer is correct.

## Cost and latency

| Metric | bf16 | awq |
|---|---:|---:|
| Completion tokens | 1,353 | 1,498 |
| Wall time (s) | 2.5 | 1.7 |
| Errors | 0 | 0 |

## Items `awq` got wrong that `bf16` got right (1)

- `inc-028` Would it be possible to add dark mode to the admin console?...

## Verdict

**REGRESSION DETECTED** — candidate is significantly worse on at least one tracked metric.
