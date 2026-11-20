# Milestone 5 Results Package

## How to reproduce

From the repository root, run:

```powershell
python milestone5_experiments.py --project-root .
```

The script performs validation based tuning and one final evaluation of each selected configuration.

## What is included

- `main_test_results.xlsx`: final test set metrics for the selected configurations.
- `selected_configurations.xlsx`: validation selected configuration for every model and horizon.
- `experiment_log.xlsx`: every validation candidate and final test run. 
- `bootstrap_confidence_intervals.xlsx`: stratified-bootstrap 95% confidence intervals.
- `main_results_figure.png`: test PR-AUC and recall comparison across horizons.
- `error_examples.xlsx`: anonymous false-negative and false-positive examples.
- `tree_depth_sensitivity.xlsx`: Decision Tree depth sensitivity records from validation.
- `public_feature_ablation.xlsx` and `public_feature_ablation_figure.png`: validation-only feature-subset comparison.
- `run_metadata.json`: reproducibility details for this run.

