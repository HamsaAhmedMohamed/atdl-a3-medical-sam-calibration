# Member 4: alternative click placement

This bundle contains updated copies of the five uploaded Python files, a new
member4_analysis.py, and CPU-only tests. Copy the Python files into your existing
project checkout, where references/Medical-SAM-Adapter, data, dependencies and
checkpoints are already set up. Review the changes with git diff before committing.
The original uploads have not been overwritten.

## Experimental choice

Agree with your group on `random_foreground` before running evaluation. It samples
three distinct foreground pixels uniformly without replacement. One/two-click
prompts are prefixes of the same three points. Points are identical for both models.
A SHA-256 hash of the prompt seed and case ID determines each random stream;
case ordering and the number of cached cases do not affect points.
This changes the entire click-placement rule, including the first point. It does
not simulate corrective clicks or clinician behaviour. The distance rule remains
the default and is unchanged. Keep sampling/split seeds unchanged (defaults: 1).

## 1. Run alternative inference

From your existing repository root:

```bash
python run_experiment.py --dataset-id refuge2 --prompt-rule random_foreground --prompt-seed 1 --run-dir runs/refuge2_random_foreground_seed1
```

The full run uses all 400 cases and the original 200/100/100 roles. It requires CUDA.
A new output directory protects the old baseline run. Prompt rule/seed enter the
cache fingerprint and are saved in metadata. Source-code changes also invalidate
caches under the existing design; do not rerun baseline inference merely to run
calibration. Do not use --case-count for the final experiment, since it changes
case selection and partition sizes.

## 2. Fit corrections and compare rules

Use shared_case_results.csv, NOT native_case_results.csv: the latter excludes the
calibration cases.

```bash
python member4_analysis.py --original runs/refuge2/shared_case_results.csv --alternative runs/refuge2_random_foreground_seed1/shared_case_results.csv --output-dir runs/member4_comparison_seed1
```

The script checks that case IDs, role assignments and model/click conditions
match across both inputs. For each model and source click count, it fits ordinary
least squares y = a*q+b using only calibration_fit cases and the clipped native
score q. It clips the resulting predictions to [0,1]. This is OLS followed by
clipping, not optimization of a loss through the clipping function.

The method is fixed in advance, so validation is not used for selection in this
implementation. If the group chooses settings on validation, freeze them before
evaluation and use the same procedure under both rules. No evaluation case is
used to fit parameters. A constant-score input falls back to the calibration mean.

Each source correction is evaluated at all three target click counts. The mean
baseline is the mean observed IoU of the source click count's calibration cases.
Original native scores are also evaluated. Neither segmentation model is trained.

## Outputs

Inside original/ and alternative/:
- fitted_corrections.csv: parameters a, b and source-condition mean baselines.
- calibration_metrics.csv: full 3x3 transfer metrics, native and mean baselines.
- calibrated_case_results.csv: individual corrected evaluation predictions.
- transfer_gap_cases.csv: per-case error(1-to-3) minus error(3-to-3).
- transfer_gap_summary.csv: mean gap and percentile 95% bootstrap interval.
- analysis_settings.json: method and uncertainty definitions.

At the comparison root:
- paired_rule_comparison.csv: alternative transfer gap minus original transfer gap,
  using paired case resampling. Positive means the alternative rule has a larger
  disadvantage from transferring the one-click correction.
- failure_example_candidates.csv: all evaluation cases ranked by absolute change
  in the transferred correction's absolute error. Use these IDs to select examples
  transparently; this file is not a set of rendered image overlays.

Intervals use 2,000 resamples of evaluation cases, with fitted corrections held
fixed. They quantify evaluation-sample uncertainty, not uncertainty from calibration
fitting or random click placement. REFUGE checkpoint-exposure risk still applies.
A single prompt seed is a limited robustness check; if repeating seeds, pre-agree
on them and report all results, not only favourable seeds. Repeated seeds are not
independent patients/cases.

## Scope

This implements the main Member 4 one-to-three transfer comparison and the full
3x3 separate-correction table. It does not implement Member 3's pooled correction
or render failure overlays. Those can reuse the newly saved predictions.

## Verification

```bash
python -m unittest discover -s . -p test_member4.py
```

Checks cover valid nested random points, seed reproducibility, original distance
placement, 200/100/100 assignment, known linear-fit recovery, constant inputs and
zero paired differences for identical data. The analysis was also run on the
uploaded baseline CSV. New GPU inference has not been run in this environment.
