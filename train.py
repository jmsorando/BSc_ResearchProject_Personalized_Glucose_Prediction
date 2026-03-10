"""
train.py
────────
Main training script. Run from repo root:

    python src/train.py                  # baseline CV only
    python src/train.py --tune           # + Optuna tuning
    python src/train.py --tune --shap    # + SHAP analysis
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

# ── Allow running as script or imported as module ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(path: Path = cfg.FEATURE_MATRIX):
    """Load feature matrix, apply quality filter, encode categoricals."""
    df = pd.read_csv(path)
    df = df[df["iauc_status"] == cfg.IAUC_STATUS].copy()
    df["sex"] = df["sex"].map({"Male": 0, "Female": 1})

    # Validate: make sure no leakage columns slipped in
    leakage_present = [c for c in cfg.LEAKAGE_COLS if c in cfg.ALL_FEATURES]
    assert not leakage_present, f"Leakage columns in feature list: {leakage_present}"

    X = df[cfg.ALL_FEATURES]
    y = df[cfg.TARGET]
    groups = df["participant_id"]

    print(f"[data]  rows={len(df):,}  features={X.shape[1]}  "
          f"participants={groups.nunique()}  "
          f"target_mean={y.mean():.1f}  target_std={y.std():.1f}")
    return X, y, groups


# ═══════════════════════════════════════════════════════════════════════════
# 2. CROSS-VALIDATED EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def cross_validate(X, y, groups, params: dict, label: str = "model") -> pd.DataFrame:
    """
    Run GroupKFold CV and return per-fold results.
    params must NOT include early_stopping_rounds if you're passing eval_set.
    """
    gkf = GroupKFold(n_splits=cfg.N_FOLDS)
    records = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = xgb.XGBRegressor(**params)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[te], y.iloc[te])],
            verbose=False,
        )
        preds = model.predict(X.iloc[te])

        records.append({
            "fold":               fold + 1,
            "label":              label,
            "MAE":                mean_absolute_error(y.iloc[te], preds),
            "RMSE":               np.sqrt(mean_squared_error(y.iloc[te], preds)),
            "R2":                 r2_score(y.iloc[te], preds),
            "n_test":             len(y.iloc[te]),
            "n_test_participants": groups.iloc[te].nunique(),
            "best_iteration":     model.best_iteration,
        })

        print(f"  fold {fold+1} | MAE={records[-1]['MAE']:6.1f}  "
              f"RMSE={records[-1]['RMSE']:6.1f}  R²={records[-1]['R2']:+.3f}  "
              f"trees={model.best_iteration}")

    res = pd.DataFrame(records)
    print(f"  {'─'*55}")
    print(f"  MEAN  | MAE={res.MAE.mean():.1f} ± {res.MAE.std():.1f}  "
          f"RMSE={res.RMSE.mean():.1f} ± {res.RMSE.std():.1f}  "
          f"R²={res.R2.mean():+.3f} ± {res.R2.std():.3f}")
    return res


# ═══════════════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════

def tune(X, y, groups, n_trials: int = cfg.OPTUNA_N_TRIALS) -> dict:
    """Bayesian hyperparameter search with Optuna. Returns best params dict."""

    gkf = GroupKFold(n_splits=cfg.N_FOLDS)

    def objective(trial):
        space = cfg.OPTUNA_SEARCH_SPACE
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",     *space["n_estimators"][1:]),
            "max_depth":        trial.suggest_int("max_depth",         *space["max_depth"][1:]),
            "learning_rate":    trial.suggest_float("learning_rate",   *space["learning_rate"][1:3], **space["learning_rate"][3]),
            "subsample":        trial.suggest_float("subsample",       *space["subsample"][1:]),
            "colsample_bytree": trial.suggest_float("colsample_bytree",*space["colsample_bytree"][1:]),
            "min_child_weight": trial.suggest_int("min_child_weight",  *space["min_child_weight"][1:]),
            "reg_alpha":        trial.suggest_float("reg_alpha",       *space["reg_alpha"][1:3], **space["reg_alpha"][3]),
            "reg_lambda":       trial.suggest_float("reg_lambda",      *space["reg_lambda"][1:3], **space["reg_lambda"][3]),
            "tree_method":      "hist",
            "random_state":     cfg.RANDOM_SEED,
            "early_stopping_rounds": 50,
        }

        fold_maes = []
        for tr, te in gkf.split(X, y, groups):
            m = xgb.XGBRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[te], y.iloc[te])],
                  verbose=False)
            fold_maes.append(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te])))
        return np.mean(fold_maes)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {
        **study.best_params,
        "tree_method":  "hist",
        "random_state": cfg.RANDOM_SEED,
        "early_stopping_rounds": 50,
    }
    print(f"\n[tune]  best MAE={study.best_value:.2f}")
    print(f"[tune]  best params: {json.dumps(study.best_params, indent=2)}")

    # Persist study and params
    joblib.dump(study, cfg.MODEL_DIR / "optuna_study.pkl")
    with open(cfg.RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"[tune]  saved → {cfg.RESULTS_DIR / 'best_params.json'}")

    return best


# ═══════════════════════════════════════════════════════════════════════════
# 4. FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════════

def train_final(X, y, params: dict) -> xgb.XGBRegressor:
    """Train on all data. Strip early_stopping_rounds (no eval_set)."""
    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**final_params)
    model.fit(X, y)

    model_path = cfg.MODEL_DIR / "final_model.ubj"
    model.save_model(model_path)
    print(f"[model] saved → {model_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 5. SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def run_shap(model, X):
    """Compute and save SHAP summary and dependence plots."""
    import shap
    import matplotlib.pyplot as plt

    print("[shap]  computing SHAP values …")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)

    # Save raw SHAP values
    shap_df = pd.DataFrame(shap_vals, columns=X.columns)
    shap_df.to_csv(cfg.RESULTS_DIR / "shap_values.csv", index=False)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, max_display=20, show=False)
    plt.tight_layout()
    p = cfg.PLOT_DIR / "shap_summary.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap]  saved → {p}")

    # Dependence plots for top 4 features by mean |SHAP|
    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
    top4 = mean_abs.nlargest(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feat in zip(axes.flat, top4):
        shap.dependence_plot(feat, shap_vals, X, ax=ax, show=False)
    plt.tight_layout()
    p = cfg.PLOT_DIR / "shap_dependence_top4.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap]  saved → {p}")

    return shap_vals


# ═══════════════════════════════════════════════════════════════════════════
# 6. ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════

def ablation(X_full, y, groups, params: dict) -> pd.DataFrame:
    """Train separate models per feature group combination."""
    import matplotlib.pyplot as plt

    feature_sets = {
        "Dc only":      cfg.DC_RAW + cfg.DC_RATIOS,
        "G only":       cfg.G_COLS,
        "Dt only":      cfg.DT_COLS,
        "Dc + G":       cfg.DC_RAW + cfg.DC_RATIOS + cfg.G_COLS,
        "Dc + Dt":      cfg.DC_RAW + cfg.DC_RATIOS + cfg.DT_COLS,
        "G + Dt":       cfg.G_COLS + cfg.DT_COLS,
        "Dc + G + Dt":  cfg.DC_RAW + cfg.DC_RATIOS + cfg.G_COLS + cfg.DT_COLS,
        "All":          cfg.ALL_FEATURES,
    }

    gkf = GroupKFold(n_splits=cfg.N_FOLDS)
    rows = []

    for label, feats in feature_sets.items():
        Xs = X_full[feats]
        r2s, maes = [], []
        for tr, te in gkf.split(Xs, y, groups):
            m = xgb.XGBRegressor(**params)
            m.fit(Xs.iloc[tr], y.iloc[tr],
                  eval_set=[(Xs.iloc[te], y.iloc[te])],
                  verbose=False)
            preds = m.predict(Xs.iloc[te])
            r2s.append(r2_score(y.iloc[te], preds))
            maes.append(mean_absolute_error(y.iloc[te], preds))
        row = {"label": label, "n_features": len(feats),
               "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
               "MAE_mean": np.mean(maes), "MAE_std": np.std(maes)}
        rows.append(row)
        print(f"  {label:14s} | n={len(feats):3d} | "
              f"R²={row['R2_mean']:+.3f} ± {row['R2_std']:.3f} | "
              f"MAE={row['MAE_mean']:.1f}")

    results = pd.DataFrame(rows)
    results.to_csv(cfg.RESULTS_DIR / "ablation.csv", index=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#4472C4" if r > 0 else "#C0504D" for r in results.R2_mean]
    ax.barh(results.label, results.R2_mean, xerr=results.R2_std,
            color=colours, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("R² (5-fold GroupKFold CV)")
    ax.set_title("Ablation Study — Feature Group Contributions")
    for i, row in results.iterrows():
        ax.text(max(row.R2_mean, 0) + 0.005, i, f"{row.R2_mean:+.3f}", va="center", fontsize=9)
    plt.tight_layout()
    p = cfg.PLOT_DIR / "ablation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ablation] saved → {p}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="iAUC XGBoost training pipeline")
    p.add_argument("--tune",     action="store_true", help="Run Optuna tuning")
    p.add_argument("--shap",     action="store_true", help="Run SHAP analysis")
    p.add_argument("--ablation", action="store_true", help="Run ablation study")
    p.add_argument("--all",      action="store_true", help="Run everything")
    p.add_argument("--data",     type=str, default=None, help="Override data path")
    return p.parse_args()


def main():
    args = parse_args()
    if args.all:
        args.tune = args.shap = args.ablation = True

    data_path = Path(args.data) if args.data else cfg.FEATURE_MATRIX
    X, y, groups = load_data(data_path)

    # ── Step 1: Baseline CV ──────────────────────────────────────────────
    print("\n" + "═"*60)
    print("BASELINE CROSS-VALIDATION")
    print("═"*60)
    baseline_results = cross_validate(X, y, groups, cfg.BASELINE_PARAMS, label="baseline")
    baseline_results.to_csv(cfg.RESULTS_DIR / "baseline_cv.csv", index=False)

    active_params = cfg.BASELINE_PARAMS

    # ── Step 2: Tuning (optional) ────────────────────────────────────────
    if args.tune:
        print("\n" + "═"*60)
        print("HYPERPARAMETER TUNING")
        print("═"*60)
        best_params = tune(X, y, groups)

        print("\nTUNED MODEL CV")
        print("─"*60)
        tuned_results = cross_validate(X, y, groups, best_params, label="tuned")
        tuned_results.to_csv(cfg.RESULTS_DIR / "tuned_cv.csv", index=False)
        active_params = best_params

    # ── Step 3: Final model ──────────────────────────────────────────────
    print("\n" + "═"*60)
    print("TRAINING FINAL MODEL (all data)")
    print("═"*60)
    final_model = train_final(X, y, active_params)

    # ── Step 4: SHAP (optional) ──────────────────────────────────────────
    if args.shap:
        print("\n" + "═"*60)
        print("SHAP ANALYSIS")
        print("═"*60)
        run_shap(final_model, X)

    # ── Step 5: Ablation (optional) ──────────────────────────────────────
    if args.ablation:
        print("\n" + "═"*60)
        print("ABLATION STUDY")
        print("═"*60)
        ablation(X, y, groups, active_params)

    print("\n✓ Pipeline complete. Outputs in:", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
