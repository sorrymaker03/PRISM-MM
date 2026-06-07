# PRISM-MM analysis code

This folder contains the analysis code used for the PRISM-MM manuscript.

## Folder structure

- `modeling/`
  Final PRISM-MM model construction and core dependencies.
  The main training script is `train_v14_multimodal_dictionary.py`.

- `benchmarking/`
  Benchmarking scripts for published perturbation-prediction methods and
  Figure 2 panel export.

- `program_analysis/`
  Program enrichment and multi-view program-correlation analysis.

- `validation/`
  Held-out bulk and external paired scRNA validation scripts.

- `biology_targets/`
  Program 6/8/9 biological interpretation and candidate target prioritization.


## Data used in this study

The analysis used a curated MM drug-perturbation transcriptomic resource with
bulk and single-cell components:

- Bulk perturbation compendium: 801 bulk transcriptomic samples across 49
  studies, 14 drugs and 25 cell-source annotations. Of these, 446 samples were
  used for PRISM-MM model construction and 355 independent samples were kept as
  held-out bulk validation data.
- Paired scRNA calibration dataset: GSE161195, including 71,820 malignant
  plasma cells from 41 patients or patient-derived samples. This dataset
  contains 41,800 control/baseline cells and 30,020 drug-treated cells, and was
  used as a single-cell calibration layer during model refinement.
- External paired scRNA validation dataset: GSE161801, including 42,440
  malignant plasma cells from 14 matched baseline-relapse pairs used for the
  main external validation.



