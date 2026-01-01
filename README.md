# theory-constrained-mental-health-ml

## Interpretable & Theory-Constrained Modeling of Mental-Health Symptoms

---

## Introduction

This repository contains a **reproducible, research-grade framework** for modeling mental-health symptom severity using **interpretable and theory-constrained machine-learning methods**. The project is motivated by a central concern in applied mental-health ML: unconstrained models can achieve strong predictive performance while violating basic psychological assumptions and obscuring failure modes.

To address this, the framework integrates **psychometric reasoning directly into the learning objective**. The model operates at the **symptom-item level** using validated clinical instruments (PHQ-9, GAD-7, PCL-5) and explicitly enforces **monotonic behavior**, ensuring that higher symptom endorsement cannot reduce predicted severity. The emphasis is on **transparent optimization, interpretability, robustness, and reproducibility**, not clinical diagnosis or deployment.

---

## Research Question

**How can mental-health symptom severity be modeled using machine-learning methods that are interpretable, theoretically valid, and stable across runs—without relying on black-box behavior?**

---

## Methods

### Model Formulation

Let `x ∈ R^p` denote item-level symptom features and `y ∈ R` a continuous severity target. The learning objective is regression under explicit monotonic inequality constraints:

min_f E[(y - f(x))^2]
s.t. ∂f(x)/∂x_j ≥ 0 for all symptom-item features j

These constraints encode the assumption that increasing symptom endorsement cannot decrease predicted severity.

### Model Class
- Gradient-boosted decision trees (CPU-efficient, histogram-based)
- Monotonic constraints applied to all symptom items
- No deep learning or GPU dependence

### Evaluation Strategy
- Repeated training across multiple random seeds (7, 21, 42)
- Performance measured via RMSE, MAE, and R²
- Residual diagnostics and prediction–truth comparisons

---

## Results

Each execution produces a **fully documented run folder** containing quantitative metrics, trained models, and diagnostic artifacts.

### Performance & Stability
- Per-seed metrics saved in `metrics.json`
- Stability summarized via **RMSE-across-seeds** plot
- Metrics demonstrate consistent performance across random initializations

### Interpretability
- **Gain-based feature importance** (Top 15 features)
- **Permutation importance** computed independently for each seed:
  - `perm_importance_seed_7.csv`
  - `perm_importance_seed_21.csv`
  - `perm_importance_seed_42.csv`
- Agreement across seeds highlights consistently influential symptom items

### Constraint Compliance
- Explicit **monotonicity audits** performed for each seed:
  - `monotonic_audit_seed_7.json`
  - `monotonic_audit_seed_21.json`
  - `monotonic_audit_seed_42.json`
- These audits quantify constraint adherence via counterfactual monotonicity sweeps

### Model Fit Diagnostics
- **Predicted vs True** scatter plots for each seed
- **Residual distribution histograms** for each seed
- Diagnostics assess calibration, symmetry, and error spread

### Psychometric Checks
- **Cronbach’s alpha** computed for PHQ-9, GAD-7, and PCL-5
- Results saved in `psychometrics_alpha.json`
- Ensures internal consistency of symptom scales used for modeling

### Data Profiling & Metadata
- Dataset summary statistics stored in `data_profile.json`
- Execution context, configuration, and environment details stored in `run_metadata.json`

---

## Output Structure

outputs/run_/20251222-184151
tables/
data_profile.json
metrics.json
psychometrics_alpha.json
monotonic_audit_seed_7.json
monotonic_audit_seed_21.json
monotonic_audit_seed_42.json
perm_importance_seed_7.csv
perm_importance_seed_21.csv
perm_importance_seed_42.csv
run_metadata.json
figures/
feature_importance_seed42.png
permutation_importance_seed7.png
permutation_importance_seed21.png
permutation_importance_seed42.png
predicted_vs_true_seed7.png
predicted_vs_true_seed21.png
predicted_vs_true_seed42.png
residual_distribution_seed7.png
residual_distribution_seed21.png
residual_distribution_seed42.png
rmse_across_seeds.png
models/
model_seed_7.joblib
model_seed_21.joblib
model_seed_42.joblib

---

## Ethics & Scope

This project is **non-diagnostic and non-clinical**. It does not aim to predict individual mental-health outcomes or real-world prevalence. The framework is designed for **methodological research, benchmarking, and education**, with all assumptions and limitations explicitly documented.

---

## Usage

### Installation
pip install numpy pandas scikit-learn xgboost matplotlib joblib

### Run
python project1_run.py
Each run generates a complete, auditable research artifact bundle under `outputs/`.

---

## Intended Audience

- Researchers studying interpretable and constrained ML
- Computational mental-health researchers
- Psychometrics and measurement-focused ML practitioners
- Students learning responsible and transparent model evaluation

---

## License & Disclaimer

This repository is provided for **research and educational use only**.  
It is **not intended for clinical diagnosis, treatment, or deployment**.






