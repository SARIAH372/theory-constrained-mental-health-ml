# Model Card — Project 1: Monotonic XGBoost for Mental-Health Severity

## Summary
Interpretable monotonic-constraint gradient boosting model trained on PHQ-9, GAD-7, and PCL-5 item-level features (reported items) to predict a continuous synthetic severity score.

## Data
Synthetic dataset reflecting validated scale structures (PHQ-9, GAD-7, PCL-5). This does not represent real-world prevalence or clinical diagnosis.

## Intended Use
Methodological demonstration of theory-constrained ML, robustness checks, and psychometric validation on synthetic mental-health scale data.

## Not Intended For
Clinical diagnosis, treatment decisions, or deployment in real healthcare contexts.

## Core Results (average across seeds)
- Main model RMSE: 0.2040
- Main model R²: 0.9945
- Ridge baseline RMSE: 0.1850

## Constraint & Robustness Evidence
- Monotonic constraints enforced; violations quantified per item (see monotonic_audit_seed_*.json).
- Stress tests include missingness injection, noise injection, and subgroup reporting bias shift (when subgroup columns exist).
- Scale-level shuffle impact reported to estimate contribution of each instrument block (PHQ/GAD/PCL).

## Outputs
- metrics.json (all metrics)
- plots (RMSE across seeds, residual diagnostics, permutation importance, feature importance)
- saved models (joblib)
- run_metadata.json & data_profile.json
