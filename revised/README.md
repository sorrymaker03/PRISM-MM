# Reviewer 1 Revision Tests

This folder contains clean, rerunnable scripts for reviewer-requested
additional analyses. The scripts are designed to write test outputs to a
separate output directory, not to overwrite manuscript figures or tables.

## Model ablation smoke test

Run from the workspace root:

```bash
.venv/bin/python codes/revised/run_reviewer1_model_ablation_test.py \
  --outdir /tmp/prism_review1_test/revised_ablation \
  --epochs 80 \
  --oof-epochs 50
```

The script evaluates four model settings:

- `full_prism_mm`: current PRISM-MM loss configuration with the cross-modal
  adapter anchor.
- `bulk_only_v9_anchor`: v9 bulk-derived anchor with scRNA reconstruction and
  cross-modal alignment disabled.
- `no_bulk_loss`: PRISM-MM structure with bulk reconstruction disabled, used as
  a direct test of whether bulk data contribute to response recovery.
- `no_alignment_loss`: PRISM-MM structure with distribution/alignment and
  study-invariance losses disabled.

Main outputs:

- `fit_metrics_summary.csv`: bulk response-recovery metrics for each ablation.
- `three_layer_program_detail.csv`: discovery bulk, held-out bulk and
  calibration scRNA program shifts.
- `three_layer_summary.csv`: number of direction-concordant and significant
  programs per ablation.

These are intended as a first-pass feasibility test. For a final revision
figure, rerun with larger `--epochs` and `--oof-epochs`.

## scRNA-only representation baseline

Run from the workspace root:

```bash
.venv/bin/python codes/revised/run_reviewer1_scrna_only_svd_baseline.py \
  --outdir /tmp/prism_review1_test/revised_ablation
```

This builds a simple scRNA-only signed SVD dictionary from paired scRNA
drug-control deltas, then evaluates the resulting programs with the same bulk
and calibration-scRNA scoring scripts. It is meant to test whether a scRNA-only
representation can recover the same cross-layer program behavior as PRISM-MM.
