"""
config.py
─────────
Single source of truth for paths, feature lists, and model defaults.
Import this everywhere. Never hardcode paths in notebooks or scripts.
"""

from pathlib import Path

# ── Repo root (works whether running locally or in Colab) ─────────────────
# In Colab after cloning:  /content/<your-repo-name>/
# Locally:                 wherever you cloned it
REPO_ROOT = Path(__file__).resolve().parent

# ── Data ──────────────────────────────────────────────────────────────────
# feature_matrix.csv lives in output/ in this repo
FEATURE_MATRIX = REPO_ROOT / "output" / "feature_matrix.csv"

# ── Outputs (created at runtime if missing) ───────────────────────────────
OUTPUT_DIR  = REPO_ROOT / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
PLOT_DIR    = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Target & filtering ────────────────────────────────────────────────────
TARGET      = "iAUC_mmol_min"
IAUC_STATUS = "ok"
RANDOM_SEED = 42
N_FOLDS     = 5

# ── Feature groups ────────────────────────────────────────────────────────
DC_RAW = [
    "CHO", "STAR", "TOTSUG", "FREE_SUGAR", "ADDED_SUGAR",
    "GLUC", "FRUCT", "SUCR", "MALT", "LACT", "GALACT", "OLIGO",
    "PROT", "FAT", "KCALS", "KJ", "ALCO", "WATER",
    "AOACFIB", "ENGFIB",
    "SATFAC", "MONOFACc", "POLYFACc", "TOTn3PFAC", "TOTn6PFAC", "FACTRANS",
    "MG", "ZN", "MN", "FE", "SE", "VITD", "CAFF",
    "totalVeg", "totalFruit",
]

DC_RATIOS = [
    "starch_fraction", "sugar_fraction", "free_sugar_fraction",
    "rapid_glucose_equiv", "intrinsic_sugar",
    "fat_cho_ratio", "protein_cho_ratio", "fibre_cho_ratio",
    "fat_sugar_ratio", "protein_sugar_ratio",
    "glycaemic_brake", "n6_n3_ratio",
]

G_COLS = [
    "baseline_glucose_mmol", "past_4h_glucose_trend",
    "past_1h_glucose_mean", "past_1h_glucose_sd", "past_1h_glucose_range",
    "mean_glucose_24h", "sd_glucose_24h", "cv_glucose_24h",
    "glucose_at_t_minus_15", "glucose_at_t_minus_30",
]

DT_COLS = [
    "past_3h_kcal", "past_3h_cho", "past_3h_sugar", "past_3h_fat", "past_3h_prot",
    "time_since_last_meal_min", "time_since_last_sig_meal_min",
    "hour_of_day",
    "is_breakfast", "is_lunch", "is_dinner", "is_snack",
]

P_COLS = [
    "sex", "n_total_meals", "n_days_tracked", "mean_daily_kcal", "mean_daily_cho",
]

# Columns that would leak postprandial information — NEVER use as features
LEAKAGE_COLS = [
    "iAUC_mmol_h", "excursion_rise_mmol", "excursion_peak_mmol",
    "peak_glucose_mmol", "time_to_peak_min", "glucose_at_120min_mmol",
]

ALL_FEATURES = DC_RAW + DC_RATIOS + G_COLS + DT_COLS + P_COLS

# ── Baseline XGBoost hyperparameters ─────────────────────────────────────
BASELINE_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=RANDOM_SEED,
    early_stopping_rounds=50,
)

# ── Optuna search space ───────────────────────────────────────────────────
OPTUNA_N_TRIALS = 100
OPTUNA_SEARCH_SPACE = {
    "n_estimators":     ("int",   200,  1000),
    "max_depth":        ("int",   3,    10),
    "learning_rate":    ("float", 0.01, 0.3,  {"log": True}),
    "subsample":        ("float", 0.6,  1.0),
    "colsample_bytree": ("float", 0.4,  1.0),
    "min_child_weight": ("int",   1,    20),
    "reg_alpha":        ("float", 1e-3, 10,   {"log": True}),
    "reg_lambda":       ("float", 1e-3, 10,   {"log": True}),
}
