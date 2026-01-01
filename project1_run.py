# ============================================================
# Project 1 — "Tier-1" Professional ML Pipeline (CPU-safe)
# Mental-health scales: PHQ-9 + GAD-7 + PCL-5 (monotonic XGBoost)
#
# Includes upgrades:
# - Run metadata + config snapshot + dataset fingerprint
# - Data profile + schema validation + NA handling + float32
# - Main monotonic XGBoost model + Ridge baseline
# - Monotonicity audit + constraint adherence score
# - Permutation importance (small sample) + plots
# - Residual diagnostics plots + predicted-vs-true
# - Scale ablation runs (fast)
# - Psychometrics: Cronbach alpha for PHQ/GAD/PCL
# - "True theta" validation correlations if present
# - Robustness tests: missingness + reporting bias shift
# - Scale-level shuffle impact (PHQ/GAD/PCL)
# - Uncertainty-lite: empirical prediction interval + coverage
#
# Notes:
# - No early_stopping_rounds (XGBoost API safe)
# - RMSE computed manually (sklearn API safe)
# - Runs remain lightweight on CPU
# ============================================================

import os
import sys
import time
import json
import math
import hashlib
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
import joblib

from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


# ----------------------------
# Auto-locate dataset
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If you want to force a filename, set it here. Otherwise it auto-picks the largest parquet in BASE_DIR.
PARQUET_FILENAME = None  # e.g. "mh_10m_sample_1m.parquet"


# ----------------------------
# Core settings (CPU-safe)
# ----------------------------
N_TRAIN = 80_000
N_VAL   = 20_000
N_TEST  = 50_000
SEEDS = [7, 21, 42]

# Limit costs for "extra rigor" tasks
PERM_IMPORTANCE_SAMPLE_N = 5000
PERM_IMPORTANCE_REPEATS = 2
MONO_SWEEP_STEPS = 25
SHIFT_NOISE_STD = 0.10      # robustness to noise (small)
MISSING_RATE = 0.05         # robustness to missingness
REPORT_BIAS_SHIFT = 1.0     # +1 shift on some items for subgroup stress test (clipped)
REPORT_BIAS_ITEMS_FRACTION = 0.25  # shift 25% of items (stress test)
PRED_INTERVAL_ALPHA = 0.10  # 90% interval


# ----------------------------
# Columns (theory-based)
# ----------------------------
PHQ_ITEMS = [f"phq{i}_rep" for i in range(1, 10)]
GAD_ITEMS = [f"gad{i}_rep" for i in range(1, 8)]
PCL_ITEMS = [f"pcl{i}_rep" for i in range(1, 21)]
FEATURES  = PHQ_ITEMS + GAD_ITEMS + PCL_ITEMS  # 36

PHQ_TOTAL = "phq9_total_rep"
GAD_TOTAL = "gad7_total_rep"
PCL_TOTAL = "pcl5_total_rep"

TARGET_COL = "y"

# Optional "true theta" / latent columns to validate against if present
THETA_CANDIDATES = [
    "theta_dep_true", "theta_anx_true", "theta_ptsd_true",
    "p_factor", "d_factor", "a_factor", "t_factor",
]

# Optional subgroup columns for fairness/shift tests if present
GROUP_CANDIDATES = ["gender", "ses", "urban", "ethnicity", "education"]


# ----------------------------
# Model configs (CPU-safe)
# ----------------------------
XGB_PARAMS = {
    "tree_method": "hist",
    "max_depth": 3,
    "learning_rate": 0.07,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 600,     # fixed; no early stopping
    "random_state": 0,
}

MONOTONE_CONSTRAINTS = tuple([1] * len(FEATURES))


# ----------------------------
# Output structure (professional run folders)
# ----------------------------
OUT_BASE = os.path.join(BASE_DIR, "outputs")


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def rmse(y_true, y_pred):
    # Compatible with all sklearn versions (no squared=)
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def find_parquet_file():
    if PARQUET_FILENAME is not None:
        p = os.path.join(BASE_DIR, PARQUET_FILENAME)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Parquet file not found: {p}")
        return p

    files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".parquet")]
    if not files:
        raise FileNotFoundError(
            f"No .parquet file found in {BASE_DIR}. "
            f"Put the parquet file next to this script or set PARQUET_FILENAME."
        )

    # pick largest
    full = [os.path.join(BASE_DIR, f) for f in files]
    full.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return full[0]


def file_fingerprint(path: str) -> dict:
    """Cheap fingerprint: size + mtime + first 1MB sha256."""
    st = os.stat(path)
    info = {"path": path, "size_bytes": st.st_size, "mtime": st.st_mtime}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    info["sha256_first1mb"] = h.hexdigest()
    return info


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """Cronbach's alpha (fast). Assumes numeric columns."""
    x = df_items.to_numpy(dtype=float)
    if x.shape[1] < 2:
        return float("nan")
    item_vars = x.var(axis=0, ddof=1)
    total = x.sum(axis=1)
    total_var = total.var(ddof=1)
    if total_var <= 1e-12:
        return float("nan")
    k = x.shape[1]
    return float((k / (k - 1)) * (1 - (item_vars.sum() / total_var)))


def pick_existing(cols, candidates):
    return [c for c in candidates if c in cols]


def main():
    # -------- run folder
    run_id = f"run_{now_stamp()}"
    run_dir = os.path.join(OUT_BASE, run_id)
    ensure_dirs(run_dir)
    ensure_dirs(os.path.join(run_dir, "figures"))
    ensure_dirs(os.path.join(run_dir, "tables"))
    ensure_dirs(os.path.join(run_dir, "models"))

    # -------- locate data
    data_path = find_parquet_file()

    # -------- metadata
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "base_dir": BASE_DIR,
        "data_path": data_path,
        "data_fingerprint": file_fingerprint(data_path),
        "settings": {
            "N_TRAIN": N_TRAIN, "N_VAL": N_VAL, "N_TEST": N_TEST,
            "SEEDS": SEEDS,
            "XGB_PARAMS": XGB_PARAMS,
            "MONO_SWEEP_STEPS": MONO_SWEEP_STEPS,
            "PERM_IMPORTANCE_SAMPLE_N": PERM_IMPORTANCE_SAMPLE_N,
            "PERM_IMPORTANCE_REPEATS": PERM_IMPORTANCE_REPEATS,
        }
    }

    print("BASE_DIR:", BASE_DIR)
    print("RUN_DIR:", run_dir)
    print("Using parquet:", data_path)

    # -------- read schema and select columns (no manual checking)
    schema_cols = pq.read_schema(data_path).names

    required = FEATURES + [PHQ_TOTAL, GAD_TOTAL, PCL_TOTAL]
    missing_req = [c for c in required if c not in schema_cols]
    if missing_req:
        raise KeyError(f"Missing required columns in parquet schema: {missing_req}")

    theta_cols = pick_existing(schema_cols, THETA_CANDIDATES)
    group_cols = pick_existing(schema_cols, GROUP_CANDIDATES)

    # We'll read required + optional if present
    read_cols = required + theta_cols + group_cols

    # -------- load data (only needed columns)
    print("Loading required columns...")
    t0 = time.time()
    table = pq.read_table(data_path, columns=read_cols)
    df = table.to_pandas()
    load_sec = round(time.time() - t0, 2)
    print("Loaded df shape:", df.shape, "| load_sec:", load_sec)

    # -------- coerce numeric required columns and handle missing
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    na_rate = df[required].isna().mean().to_dict()
    medians = df[required].median(numeric_only=True)
    df[required] = df[required].fillna(medians)

    # float32 for speed/memory
    df[FEATURES] = df[FEATURES].astype("float32")

    # -------- build target
    df[TARGET_COL] = zscore(df[PHQ_TOTAL]) + zscore(df[GAD_TOTAL]) + zscore(df[PCL_TOTAL])

    # -------- data profile
    profile = {
        "loaded_shape": df.shape,
        "load_seconds": load_sec,
        "na_rate_required": na_rate,
        "totals_summary": {
            PHQ_TOTAL: df[PHQ_TOTAL].describe().to_dict(),
            GAD_TOTAL: df[GAD_TOTAL].describe().to_dict(),
            PCL_TOTAL: df[PCL_TOTAL].describe().to_dict(),
        },
        "optional_theta_cols_present": theta_cols,
        "optional_group_cols_present": group_cols,
    }
    save_json(profile, os.path.join(run_dir, "tables", "data_profile.json"))
    save_json(metadata, os.path.join(run_dir, "tables", "run_metadata.json"))

    # -------- psychometrics (alpha) on a small subsample for speed
    alpha_sample_n = min(50_000, len(df))
    alpha_idx = np.random.default_rng(123).choice(len(df), size=alpha_sample_n, replace=False)
    alpha_df = df.iloc[alpha_idx]

    alpha_report = {
        "alpha_sample_n": int(alpha_sample_n),
        "cronbach_alpha_phq": cronbach_alpha(alpha_df[PHQ_ITEMS]),
        "cronbach_alpha_gad": cronbach_alpha(alpha_df[GAD_ITEMS]),
        "cronbach_alpha_pcl": cronbach_alpha(alpha_df[PCL_ITEMS]),
    }
    save_json(alpha_report, os.path.join(run_dir, "tables", "psychometrics_alpha.json"))

    # -------- training loop
    results = []
    rmse_list = []
    last_model = None
    last_seed = None

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        rng = np.random.default_rng(seed)

        n_total = N_TRAIN + N_VAL + N_TEST
        idx = rng.choice(len(df), size=n_total, replace=False)
        sub = df.iloc[idx].copy()

        perm = rng.permutation(n_total)
        train = sub.iloc[perm[:N_TRAIN]]
        val   = sub.iloc[perm[N_TRAIN:N_TRAIN + N_VAL]]
        test  = sub.iloc[perm[N_TRAIN + N_VAL:]]

        X_train = train[FEATURES].astype("float32")
        y_train = train[TARGET_COL].astype(float)

        X_val   = val[FEATURES].astype("float32")
        y_val   = val[TARGET_COL].astype(float)

        X_test  = test[FEATURES].astype("float32")
        y_test  = test[TARGET_COL].astype(float)

        # ---- main model
        model = XGBRegressor(**XGB_PARAMS, monotone_constraints=MONOTONE_CONSTRAINTS)

        t_train = time.time()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_sec = round(time.time() - t_train, 2)

        preds = model.predict(X_test)
        main_rmse = rmse(y_test, preds)
        main_mae  = float(mean_absolute_error(y_test, preds))
        main_r2   = float(r2_score(y_test, preds))

        # ---- baseline
        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        ridge_rmse = rmse(y_test, ridge_pred)
        ridge_mae  = float(mean_absolute_error(y_test, ridge_pred))
        ridge_r2   = float(r2_score(y_test, ridge_pred))

        print(f"Main:  RMSE={main_rmse:.4f}  R2={main_r2:.4f}  (train_sec={train_sec})")
        print(f"Ridge: RMSE={ridge_rmse:.4f}  R2={ridge_r2:.4f}")

        seed_result = {
            "seed": seed,
            "train_sec": float(train_sec),
            "main_rmse": main_rmse,
            "main_mae": main_mae,
            "main_r2": main_r2,
            "ridge_rmse": ridge_rmse,
            "ridge_mae": ridge_mae,
            "ridge_r2": ridge_r2,
        }

        # ---- monotonic audit (cheap proof)
        med = X_test[FEATURES].median(numeric_only=True)
        audit = []
        total_viol = 0
        zero_viol = 0

        for f in FEATURES:
            grid = np.linspace(float(X_test[f].min()), float(X_test[f].max()), MONO_SWEEP_STEPS)
            Xg = pd.DataFrame([med.values] * len(grid), columns=FEATURES).astype("float32")
            Xg[f] = grid.astype("float32")
            p = model.predict(Xg)
            diffs = np.diff(p)
            viol = int((diffs < -1e-8).sum())
            total_viol += viol
            if viol == 0:
                zero_viol += 1
            audit.append({"feature": f, "violations": viol, "steps": int(len(diffs))})

        seed_result["monotone_total_violations"] = int(total_viol)
        seed_result["monotone_features_zero_violation"] = int(zero_viol)
        seed_result["monotone_zero_violation_rate"] = float(zero_viol / len(FEATURES))

        save_json(audit, os.path.join(run_dir, "tables", f"monotonic_audit_seed_{seed}.json"))

        # ---- permutation importance (bounded)
        sample_n = min(PERM_IMPORTANCE_SAMPLE_N, len(X_test))
        pi_idx = np.random.default_rng(seed).choice(len(X_test), size=sample_n, replace=False)
        X_pi = X_test.iloc[pi_idx]
        y_pi = y_test.iloc[pi_idx]

        perm_imp = permutation_importance(
            model, X_pi, y_pi,
            n_repeats=PERM_IMPORTANCE_REPEATS,
            random_state=seed,
            scoring="neg_mean_squared_error"  # safe across sklearn versions
        )
        pi_df = pd.DataFrame({
            "feature": FEATURES,
            "importance_mean": perm_imp.importances_mean,
            "importance_std": perm_imp.importances_std
        }).sort_values("importance_mean", ascending=False)

        pi_csv = os.path.join(run_dir, "tables", f"perm_importance_seed_{seed}.csv")
        pi_df.to_csv(pi_csv, index=False)

        top_pi = pi_df.head(15).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top_pi["feature"], top_pi["importance_mean"])
        plt.xlabel("Permutation Importance (Δ -MSE)")
        plt.title(f"Top 15 Permutation Importances (seed {seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "figures", f"perm_importance_top15_seed_{seed}.png"), dpi=150)

        # ---- residual diagnostics
        residuals = (y_test.to_numpy() - preds)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, preds, s=5)
        plt.xlabel("True y")
        plt.ylabel("Predicted y")
        plt.title(f"Predicted vs True (seed {seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "figures", f"pred_vs_true_seed_{seed}.png"), dpi=150)

        plt.figure(figsize=(7, 4))
        plt.hist(residuals, bins=50)
        plt.xlabel("Residual (y_true - y_pred)")
        plt.ylabel("Count")
        plt.title(f"Residual Distribution (seed {seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "figures", f"residual_hist_seed_{seed}.png"), dpi=150)

        # ---- scale ablation (fast mini-runs)
        def quick_fit(cols, tag):
            m = XGBRegressor(
                tree_method="hist",
                max_depth=3,
                learning_rate=0.07,
                subsample=0.8,
                colsample_bytree=0.8,
                n_estimators=200,
                random_state=seed
            )
            m.fit(X_train[cols], y_train, eval_set=[(X_val[cols], y_val)], verbose=False)
            p = m.predict(X_test[cols])
            seed_result[f"ablation_rmse_{tag}"] = rmse(y_test, p)
            seed_result[f"ablation_r2_{tag}"] = float(r2_score(y_test, p))

        quick_fit(PHQ_ITEMS, "phq")
        quick_fit(GAD_ITEMS, "gad")
        quick_fit(PCL_ITEMS, "pcl")
        quick_fit(FEATURES,  "all")

        # ---- theta validation (if present)
        if theta_cols:
            corr_report = {}
            for tc in theta_cols:
                # correlate predictions with theta on test (spearman & pearson)
                tvals = test[tc]
                if pd.api.types.is_numeric_dtype(tvals):
                    corr_report[f"{tc}_pearson"] = float(pd.Series(preds).corr(tvals, method="pearson"))
                    corr_report[f"{tc}_spearman"] = float(pd.Series(preds).corr(tvals, method="spearman"))
            seed_result["theta_validation"] = corr_report

        # ---- robustness: missingness (no retraining)
        X_miss = X_test.copy()
        miss_rng = np.random.default_rng(seed + 1000)
        mask = miss_rng.random(X_miss.shape) < MISSING_RATE
        X_miss = X_miss.mask(mask)
        # median impute
        X_miss = X_miss.fillna(X_test.median(numeric_only=True))
        pred_miss = model.predict(X_miss)
        seed_result["robust_rmse_missing_5pct"] = rmse(y_test, pred_miss)

        # ---- robustness: small noise
        X_noise = X_test.to_numpy(dtype=np.float32)
        noise = np.random.default_rng(seed + 2000).normal(0, SHIFT_NOISE_STD, size=X_noise.shape).astype(np.float32)
        X_noise = X_noise + noise
        X_noise = pd.DataFrame(X_noise, columns=FEATURES)
        pred_noise = model.predict(X_noise)
        seed_result["robust_rmse_noise"] = rmse(y_test, pred_noise)

        # ---- reporting bias shift test (if group exists)
        group_used = None
        if group_cols:
            # pick first available group col
            group_used = group_cols[0]
        if group_used is not None:
            g = test[group_used]
            # pick a "subgroup": the most frequent category value
            try:
                subgroup_value = g.value_counts().index[0]
                subgroup_mask = (g == subgroup_value).to_numpy()
            except Exception:
                subgroup_mask = None

            if subgroup_mask is not None and subgroup_mask.sum() > 50:
                X_shift = X_test.copy()

                # choose subset of items to shift
                item_count = len(FEATURES)
                k = max(1, int(item_count * REPORT_BIAS_ITEMS_FRACTION))
                shift_items = list(np.random.default_rng(seed + 3000).choice(FEATURES, size=k, replace=False))

                # shift + clip to observed min/max
                for col in shift_items:
                    mn = float(X_test[col].min())
                    mx = float(X_test[col].max())
                    X_shift.loc[subgroup_mask, col] = np.clip(
                        X_shift.loc[subgroup_mask, col].to_numpy(dtype=float) + REPORT_BIAS_SHIFT,
                        mn, mx
                    ).astype("float32")

                pred_shift = model.predict(X_shift)
                seed_result["report_bias_shift_group_col"] = str(group_used)
                seed_result["report_bias_shift_group_value"] = str(subgroup_value)
                seed_result["robust_rmse_reporting_bias_shift"] = rmse(y_test, pred_shift)

        # ---- scale-level shuffle impact (no retraining)
        def shuffle_block(cols, tag):
            Xb = X_test.copy()
            rr = np.random.default_rng(seed + 4000 + hash(tag) % 1000)
            for c in cols:
                Xb[c] = rr.permutation(Xb[c].to_numpy())
            pb = model.predict(Xb)
            seed_result[f"shuffle_block_rmse_{tag}"] = rmse(y_test, pb)

        shuffle_block(PHQ_ITEMS, "phq")
        shuffle_block(GAD_ITEMS, "gad")
        shuffle_block(PCL_ITEMS, "pcl")

        # ---- uncertainty-lite: prediction interval from validation residuals
        val_pred = model.predict(X_val)
        val_resid = (y_val.to_numpy() - val_pred)
        lo = float(np.quantile(val_resid, PRED_INTERVAL_ALPHA / 2))
        hi = float(np.quantile(val_resid, 1 - PRED_INTERVAL_ALPHA / 2))

        # interval coverage on test
        lower = preds + lo
        upper = preds + hi
        yt = y_test.to_numpy()
        covered = ((yt >= lower) & (yt <= upper)).mean()

        seed_result["pred_interval_resid_q_lo"] = lo
        seed_result["pred_interval_resid_q_hi"] = hi
        seed_result["pred_interval_coverage_test"] = float(covered)

        # save model
        joblib.dump(model, os.path.join(run_dir, "models", f"model_seed_{seed}.joblib"))

        results.append(seed_result)
        rmse_list.append(main_rmse)

        last_model = model
        last_seed = seed

    # -------- summary outputs
    metrics_path = os.path.join(run_dir, "tables", "metrics.json")
    save_json(results, metrics_path)

    # RMSE across seeds plot
    plt.figure()
    plt.plot([r["seed"] for r in results], [r["main_rmse"] for r in results], marker="o")
    plt.xlabel("Seed")
    plt.ylabel("RMSE")
    plt.title("Project 1 — RMSE across seeds (Monotone XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "figures", "rmse_across_seeds.png"), dpi=150)

    # Feature importance plot (built-in importance, quick)
    if last_model is not None:
        imp = last_model.feature_importances_
        imp_df = pd.DataFrame({"feature": FEATURES, "importance": imp}).sort_values("importance", ascending=False)
        top = imp_df.head(15).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("XGBoost Feature Importance")
        plt.title(f"Top 15 Feature Importances (seed {last_seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "figures", "feature_importance_top15.png"), dpi=150)

    # -------- model card
    avg_rmse = float(np.mean([r["main_rmse"] for r in results]))
    avg_r2 = float(np.mean([r["main_r2"] for r in results]))
    avg_ridge_rmse = float(np.mean([r["ridge_rmse"] for r in results]))

    model_card = f"""# Model Card — Project 1: Monotonic XGBoost for Mental-Health Severity

## Summary
Interpretable monotonic-constraint gradient boosting model trained on PHQ-9, GAD-7, and PCL-5 item-level features (reported items) to predict a continuous synthetic severity score.

## Data
Synthetic dataset reflecting validated scale structures (PHQ-9, GAD-7, PCL-5). This does not represent real-world prevalence or clinical diagnosis.

## Intended Use
Methodological demonstration of theory-constrained ML, robustness checks, and psychometric validation on synthetic mental-health scale data.

## Not Intended For
Clinical diagnosis, treatment decisions, or deployment in real healthcare contexts.

## Core Results (average across seeds)
- Main model RMSE: {avg_rmse:.4f}
- Main model R²: {avg_r2:.4f}
- Ridge baseline RMSE: {avg_ridge_rmse:.4f}

## Constraint & Robustness Evidence
- Monotonic constraints enforced; violations quantified per item (see monotonic_audit_seed_*.json).
- Stress tests include missingness injection, noise injection, and subgroup reporting bias shift (when subgroup columns exist).
- Scale-level shuffle impact reported to estimate contribution of each instrument block (PHQ/GAD/PCL).

## Outputs
- metrics.json (all metrics)
- plots (RMSE across seeds, residual diagnostics, permutation importance, feature importance)
- saved models (joblib)
- run_metadata.json & data_profile.json
"""
    with open(os.path.join(run_dir, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.write(model_card)

    print("\n✅ DONE!")
    print("Run folder:", run_dir)
    print("Metrics:", metrics_path)
    print("Key figures are in:", os.path.join(run_dir, "figures"))


if __name__ == "__main__":
    main()




