#!/usr/bin/env python3
"""
Build XGBoost feature matrix for iAUC prediction.

Each row = one glucose excursion. Target = 2h postprandial iAUC.
Features: glycaemic context (G), diet composition (Dc), diet temporal (Dt).

Output: output/feature_matrix.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────
BASE         = Path(r"C:\Users\Jose Miguel Sorando\Documents\RP Cleaning 5")
CGM_DIR      = BASE / "source" / "cgm_data"
MEAL_FILE    = BASE / "output" / "corrected_meal_times_ALL.csv"
EXTRACT_FILE = BASE / "output" / "patient_extract1602_realigned.csv"
MATCH_FILE   = BASE / "source" / "MyFood24 ID Matched(Sheet1).csv"
SOURCE_FILE  = BASE / "source" / "patient_extract1602.csv"
OUTPUT_FILE  = BASE / "output" / "feature_matrix.csv"

WINDOW_MIN     = 120
EXPECTED_READS = 25
SENTINEL_LOW   = 2.2
SENTINEL_HIGH  = 22.2
MAX_NADIR_SNAP = 10     # max minutes between nadir and nearest CGM reading
GAP_FLAG_MIN   = 15     # flag events with CGM gaps exceeding this

# Nutrient columns to extract from patient_extract
NUTRIENT_COLS = [
    "totalVeg", "totalFruit", "WATER", "PROT", "FAT", "CHO", "KCALS", "KJ",
    "STAR", "OLIGO", "TOTSUG", "GLUC", "GALACT", "FRUCT", "SUCR", "MALT",
    "LACT", "ALCO", "ENGFIB", "AOACFIB",
    "SATFAC", "TOTn6PFAC", "TOTn3PFAC", "MONOFACc", "POLYFACc", "FACTRANS",
    "MG", "FE", "ZN", "MN", "SE", "VITD",
    "FREE_SUGAR", "ADDED_SUGAR", "CAFF",
]


# ── Helpers ───────────────────────────────────────────────────────
def _parse_tz_offset(s):
    """Parse '+01:00' or '-05:30' into a timedelta."""
    try:
        s = str(s).strip()
        sign = 1 if s[0] == "+" else -1
        pts = s[1:].split(":")
        return timedelta(hours=sign * int(pts[0]),
                         minutes=sign * int(pts[1]) if len(pts) > 1 else 0)
    except Exception:
        return timedelta(0)


def _load_cgm(path):
    """Load and clean a single CGM file. Returns DataFrame with ts_utc, glucose."""
    cgm = pd.read_csv(path, low_memory=False)
    if "event_type" in cgm.columns:
        cgm = cgm[cgm["event_type"] == "EGV"].copy()
    cgm["ts_utc"] = pd.to_datetime(cgm["isoDate"], utc=True, errors="coerce")
    cgm["glucose"] = cgm["glucose"].replace({"Low": SENTINEL_LOW, "High": SENTINEL_HIGH})
    cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")
    cgm = cgm.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    return cgm[["ts_utc", "glucose"]].copy()


def _compute_iauc(cgm_window, g0):
    """Trapezoidal iAUC (positive-only) over a CGM window. Returns mmol·min/L."""
    ts = cgm_window["ts_utc"].values
    gl = cgm_window["glucose"].values
    iauc = 0.0
    for i in range(len(gl) - 1):
        if np.isnan(gl[i]) or np.isnan(gl[i + 1]):
            continue
        d0 = max(gl[i] - g0, 0.0)
        d1 = max(gl[i + 1] - g0, 0.0)
        dt = (ts[i + 1] - ts[i]) / np.timedelta64(1, "m")
        iauc += 0.5 * (d0 + d1) * dt
    return iauc


def _nearest_glucose(cgm, target_utc, max_snap_min=3):
    """Find nearest CGM reading to target_utc. Returns glucose or NaN."""
    if cgm is None or len(cgm) == 0:
        return np.nan
    diffs = (cgm["ts_utc"] - target_utc).abs()
    idx = diffs.idxmin()
    snap = diffs.loc[idx].total_seconds() / 60
    if snap > max_snap_min:
        return np.nan
    return cgm.loc[idx, "glucose"]


def _normalise_date(d):
    """Normalise various date formats to YYYY-MM-DD string."""
    try:
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    except Exception:
        return None


# ── Step 1: Nutrient Enrichment ──────────────────────────────────
def step1_nutrient_enrichment(meals, extract):
    """Join patient_extract items to meal events to recover full nutrient panel."""
    print("\n  Step 1: Nutrient Enrichment")
    print("  " + "-" * 40)

    # Normalise dates
    extract["_date_norm"] = extract["Date"].apply(_normalise_date)
    extract["_pid"] = extract["Patient Id"]

    # Parse time consumed for disambiguation
    extract["_time_consumed"] = pd.to_datetime(
        extract["Time consumed at"], format="%H:%M", errors="coerce"
    )

    # Initialise nutrient columns on meals
    for col in NUTRIENT_COLS:
        meals[col] = np.nan

    n_success = 0
    n_fail = 0
    n_mismatch = 0
    mismatches = []

    for idx in meals.index:
        pid = meals.loc[idx, "myfood24_id"]
        date_str = _normalise_date(meals.loc[idx, "date"])
        food_str = str(meals.loc[idx, "food_items"])
        if food_str == "nan" or not food_str.strip():
            n_fail += 1
            continue

        food_names = [f.strip() for f in food_str.split(";")]

        # Filter extract to same patient + same date
        day_items = extract[(extract["_pid"] == pid) &
                            (extract["_date_norm"] == date_str)].copy()
        if len(day_items) == 0:
            n_fail += 1
            continue

        # Parse event reported_time for disambiguation
        try:
            evt_time = pd.Timestamp(meals.loc[idx, "reported_time"])
            evt_minutes = evt_time.hour * 60 + evt_time.minute
        except Exception:
            evt_minutes = None

        # Track which extract rows have been used (by original index)
        used_indices = set()
        matched_rows = []

        for fname in food_names:
            fname_clean = fname.strip()
            if not fname_clean:
                continue

            candidates = day_items[
                (day_items["Food name"] == fname_clean) &
                (~day_items.index.isin(used_indices))
            ]

            if len(candidates) == 0:
                continue

            if len(candidates) == 1:
                best = candidates.index[0]
            else:
                # Disambiguate by time proximity
                if evt_minutes is not None and candidates["_time_consumed"].notna().any():
                    cand_minutes = candidates["_time_consumed"].dt.hour * 60 + \
                                   candidates["_time_consumed"].dt.minute
                    time_diffs = (cand_minutes - evt_minutes).abs()
                    best = time_diffs.idxmin()
                else:
                    best = candidates.index[0]

            used_indices.add(best)
            matched_rows.append(best)

        if len(matched_rows) == 0:
            n_fail += 1
            continue

        # Sum nutrients from matched rows
        for col in NUTRIENT_COLS:
            if col in extract.columns:
                vals = pd.to_numeric(extract.loc[matched_rows, col], errors="coerce")
                meals.loc[idx, col] = round(vals.sum(), 4)

        n_success += 1

        # Verification: CHO should match total_CHO within ±5g
        enriched_cho = meals.loc[idx, "CHO"]
        orig_cho = pd.to_numeric(meals.loc[idx, "total_CHO"], errors="coerce")
        if not np.isnan(enriched_cho) and not np.isnan(orig_cho):
            if abs(enriched_cho - orig_cho) > 5:
                n_mismatch += 1
                if len(mismatches) < 5:
                    mismatches.append((meals.loc[idx, "event_id"],
                                      enriched_cho, orig_cho))

    total = n_success + n_fail
    pct = 100 * n_success / total if total > 0 else 0
    print(f"    Matched: {n_success}/{total} events ({pct:.1f}%)")
    print(f"    CHO mismatches (>5g): {n_mismatch}")
    if mismatches:
        for eid, ec, oc in mismatches:
            print(f"      {eid}: enriched={ec:.1f}g vs original={oc:.1f}g")

    return meals


# ── Step 2: Excursion-Level Aggregation ──────────────────────────
def step2_aggregate_stacking(meals):
    """Group by (participant_id, excursion_id), sum nutrients for stacked meals."""
    print("\n  Step 2: Excursion-Level Aggregation")
    print("  " + "-" * 40)

    # Separate events with and without excursion_id
    has_exc = meals[meals["excursion_id"].notna() & (meals["excursion_id"] != "")].copy()
    no_exc = meals[meals["excursion_id"].isna() | (meals["excursion_id"] == "")].copy()

    # For events with excursion_id, group by (participant_id, excursion_id)
    grouped_rows = []
    n_stacked_groups = 0

    for (pid, eid), grp in has_exc.groupby(["participant_id", "excursion_id"]):
        anchor = grp[grp["match_type"] == "anchor"]
        if len(anchor) == 0:
            anchor = grp.iloc[[0]]
        else:
            anchor = anchor.iloc[[0]]

        row = anchor.iloc[0].copy()

        if len(grp) > 1:
            n_stacked_groups += 1
            # Sum nutrient columns
            for col in NUTRIENT_COLS:
                if col in grp.columns:
                    vals = pd.to_numeric(grp[col], errors="coerce")
                    row[col] = round(vals.sum(), 4)

            # Concatenate food items, meal labels, event ids
            row["food_items"] = "; ".join(grp["food_items"].dropna().astype(str))
            row["meal_label"] = " + ".join(grp["meal_label"].dropna().astype(str))
            row["event_id"] = "; ".join(grp["event_id"].dropna().astype(str))
            row["n_items"] = pd.to_numeric(grp["n_items"], errors="coerce").sum()

        row["n_meals_in_excursion"] = len(grp)
        grouped_rows.append(row)

    df_exc = pd.DataFrame(grouped_rows)

    # For unmatched events, each is its own row
    no_exc = no_exc.copy()
    no_exc["n_meals_in_excursion"] = 1

    result = pd.concat([df_exc, no_exc], ignore_index=True)

    print(f"    Excursions with stacking: {n_stacked_groups}")
    print(f"    Events with excursion_id: {len(has_exc)} -> {len(df_exc)} rows")
    print(f"    Events without excursion_id: {len(no_exc)} rows")
    print(f"    Total rows: {len(result)}")

    return result


# ── Step 3: iAUC Computation ────────────────────────────────────
def step3_compute_iauc(df, cgm_cache, cgm_files):
    """Compute iAUC and CGM quality metrics for each row with a nadir_time."""
    print("\n  Step 3: iAUC Computation")
    print("  " + "-" * 40)

    iauc_out_cols = ["baseline_glucose_mmol", "iAUC_mmol_min", "iAUC_mmol_h",
                     "peak_glucose_mmol", "time_to_peak_min", "glucose_at_120min_mmol",
                     "n_readings", "pct_coverage", "max_gap_min", "iauc_status"]
    for c in iauc_out_cols:
        if c not in df.columns:
            df[c] = np.nan
    if "iauc_status" not in df.columns or df["iauc_status"].dtype != object:
        df["iauc_status"] = ""

    status_counts = Counter()
    pids = df["participant_id"].unique()

    for pi, pid in enumerate(sorted(pids)):
        sub = df[df["participant_id"] == pid]

        # Load CGM
        if pid not in cgm_cache:
            if pid in cgm_files:
                cgm_cache[pid] = _load_cgm(cgm_files[pid])
            else:
                cgm_cache[pid] = None
        cgm = cgm_cache[pid]

        for idx in sub.index:
            nadir = df.loc[idx, "nadir_time"]

            if pd.isna(nadir) or str(nadir).strip() == "":
                conf = str(df.loc[idx, "confidence"])
                df.loc[idx, "iauc_status"] = conf
                status_counts[conf] += 1
                continue

            if cgm is None or len(cgm) == 0:
                df.loc[idx, "iauc_status"] = "no_cgm_file"
                status_counts["no_cgm_file"] += 1
                continue

            # Convert nadir from local to UTC
            tz_off = _parse_tz_offset(df.loc[idx, "tz_offset"])
            nadir_utc = pd.Timestamp(nadir).tz_localize(None) - tz_off
            nadir_utc = nadir_utc.tz_localize("UTC")

            # Find nearest CGM reading to nadir
            diffs = (cgm["ts_utc"] - nadir_utc).abs()
            nearest_idx = diffs.idxmin()
            snap_min = diffs.loc[nearest_idx].total_seconds() / 60

            if snap_min > MAX_NADIR_SNAP:
                df.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue

            g0 = cgm.loc[nearest_idx, "glucose"]
            if np.isnan(g0):
                df.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue

            # Extract 2h window
            window_end = nadir_utc + pd.Timedelta(minutes=WINDOW_MIN)
            mask = (cgm["ts_utc"] >= nadir_utc - pd.Timedelta(seconds=30)) & \
                   (cgm["ts_utc"] <= window_end + pd.Timedelta(seconds=30))
            window = cgm.loc[mask].copy()

            if len(window) < 3:
                df.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue

            n_readings = len(window)
            pct_cov = n_readings / EXPECTED_READS
            gaps = window["ts_utc"].diff().dt.total_seconds() / 60
            max_gap = gaps.max() if len(gaps) > 1 else 0.0

            iauc = _compute_iauc(window, g0)

            # Peak in window
            valid_gl = window.dropna(subset=["glucose"])
            if len(valid_gl) > 0:
                peak_idx = valid_gl["glucose"].idxmax()
                peak_gl = valid_gl.loc[peak_idx, "glucose"]
                peak_ts = valid_gl.loc[peak_idx, "ts_utc"]
                ttp = (peak_ts - nadir_utc).total_seconds() / 60
            else:
                peak_gl = np.nan
                ttp = np.nan

            # Glucose at 120 min
            end_diffs = (window["ts_utc"] - window_end).abs()
            end_idx = end_diffs.idxmin()
            end_snap = end_diffs.loc[end_idx].total_seconds() / 60
            gl_120 = window.loc[end_idx, "glucose"] if end_snap <= 10 else np.nan

            status = "gap_too_large" if max_gap > GAP_FLAG_MIN else "ok"

            df.loc[idx, "baseline_glucose_mmol"] = round(g0, 4)
            df.loc[idx, "iAUC_mmol_min"] = round(iauc, 4)
            df.loc[idx, "iAUC_mmol_h"] = round(iauc / 60, 4)
            df.loc[idx, "peak_glucose_mmol"] = round(peak_gl, 4)
            df.loc[idx, "time_to_peak_min"] = round(ttp, 1)
            df.loc[idx, "glucose_at_120min_mmol"] = (
                round(gl_120, 4) if not np.isnan(gl_120) else np.nan
            )
            df.loc[idx, "n_readings"] = int(n_readings)
            df.loc[idx, "pct_coverage"] = round(pct_cov, 4)
            df.loc[idx, "max_gap_min"] = round(max_gap, 1)
            df.loc[idx, "iauc_status"] = status
            status_counts[status] += 1

        if (pi + 1) % 10 == 0 or pi + 1 == len(pids):
            print(f"    [{pi+1}/{len(pids)}] {pid}")

    computed = (status_counts.get("ok", 0) + status_counts.get("gap_too_large", 0))
    print(f"    Computed iAUC for {computed} rows")
    for s, n in status_counts.most_common():
        print(f"      {s:30s} {n:>5d}")

    return df, cgm_cache


# ── Step 4: Glycaemic Features ──────────────────────────────────
def step4_glycaemic_features(df, cgm_cache):
    """Compute CGM-derived glycaemic context features for each row."""
    print("\n  Step 4: Glycaemic Features (G)")
    print("  " + "-" * 40)

    g_cols = ["past_4h_glucose_trend",
              "past_1h_glucose_mean", "past_1h_glucose_sd", "past_1h_glucose_range",
              "mean_glucose_24h", "sd_glucose_24h", "cv_glucose_24h",
              "glucose_at_t_minus_15", "glucose_at_t_minus_30"]
    for c in g_cols:
        df[c] = np.nan

    n_computed = 0
    pids = df["participant_id"].unique()

    for pid in sorted(pids):
        cgm = cgm_cache.get(pid)
        if cgm is None or len(cgm) == 0:
            continue

        sub_idx = df[df["participant_id"] == pid].index

        for idx in sub_idx:
            nadir = df.loc[idx, "nadir_time"]
            if pd.isna(nadir) or str(nadir).strip() == "":
                continue

            tz_off = _parse_tz_offset(df.loc[idx, "tz_offset"])
            t0_utc = pd.Timestamp(nadir).tz_localize(None) - tz_off
            t0_utc = t0_utc.tz_localize("UTC")

            # 4b. Past 4h glucose trend (linear regression slope)
            mask_4h = (cgm["ts_utc"] >= t0_utc - pd.Timedelta(hours=4)) & \
                      (cgm["ts_utc"] <= t0_utc)
            window_4h = cgm.loc[mask_4h].dropna(subset=["glucose"])

            if len(window_4h) >= 6:
                x = (window_4h["ts_utc"] - t0_utc).dt.total_seconds().values / 3600
                y = window_4h["glucose"].values
                # Simple OLS: slope = cov(x,y) / var(x)
                x_mean = x.mean()
                y_mean = y.mean()
                var_x = ((x - x_mean) ** 2).sum()
                if var_x > 0:
                    slope = ((x - x_mean) * (y - y_mean)).sum() / var_x
                    df.loc[idx, "past_4h_glucose_trend"] = round(slope, 4)

            # 4c. Past 1h glucose statistics
            mask_1h = (cgm["ts_utc"] >= t0_utc - pd.Timedelta(hours=1)) & \
                      (cgm["ts_utc"] <= t0_utc)
            window_1h = cgm.loc[mask_1h].dropna(subset=["glucose"])

            if len(window_1h) >= 2:
                gl = window_1h["glucose"]
                df.loc[idx, "past_1h_glucose_mean"] = round(gl.mean(), 4)
                df.loc[idx, "past_1h_glucose_sd"] = round(gl.std(), 4)
                df.loc[idx, "past_1h_glucose_range"] = round(gl.max() - gl.min(), 4)

            # 4d. Past 24h cumulative glucose metrics
            mask_24h = (cgm["ts_utc"] >= t0_utc - pd.Timedelta(hours=24)) & \
                       (cgm["ts_utc"] <= t0_utc)
            window_24h = cgm.loc[mask_24h].dropna(subset=["glucose"])

            if len(window_24h) >= 12:
                gl24 = window_24h["glucose"]
                m24 = gl24.mean()
                s24 = gl24.std()
                df.loc[idx, "mean_glucose_24h"] = round(m24, 4)
                df.loc[idx, "sd_glucose_24h"] = round(s24, 4)
                df.loc[idx, "cv_glucose_24h"] = round(s24 / m24, 4) if m24 > 0 else np.nan

            # 4e. Glucose at fixed pre-meal timepoints
            t_minus_15 = t0_utc - pd.Timedelta(minutes=15)
            t_minus_30 = t0_utc - pd.Timedelta(minutes=30)

            df.loc[idx, "glucose_at_t_minus_15"] = _nearest_glucose(cgm, t_minus_15, 3)
            df.loc[idx, "glucose_at_t_minus_30"] = _nearest_glucose(cgm, t_minus_30, 3)

            n_computed += 1

    print(f"    Computed glycaemic features for {n_computed} rows")
    for c in g_cols:
        pct = 100 * df[c].notna().sum() / len(df) if len(df) > 0 else 0
        print(f"      {c:30s} {pct:5.1f}% non-null")

    return df


# ── Step 5: Diet Temporal Features ──────────────────────────────
def step5_diet_temporal(df, all_meals):
    """Compute diet temporal context features from the meal timeline."""
    print("\n  Step 5: Diet Temporal Features (Dt)")
    print("  " + "-" * 40)

    dt_cols = ["past_3h_kcal", "past_3h_cho", "past_3h_sugar",
               "past_3h_fat", "past_3h_prot",
               "time_since_last_meal_min", "time_since_last_sig_meal_min",
               "hour_of_day",
               "is_breakfast", "is_lunch", "is_dinner", "is_snack"]
    for c in dt_cols:
        df[c] = np.nan

    # Parse corrected_time as datetime in all_meals for temporal lookups
    all_meals = all_meals.copy()
    all_meals["_ct"] = pd.to_datetime(all_meals["corrected_time"], errors="coerce")
    all_meals["_cho_val"] = pd.to_numeric(all_meals["total_CHO"], errors="coerce")
    all_meals["_kcal_val"] = pd.to_numeric(all_meals["total_KCALS"], errors="coerce")
    all_meals["_totsug_val"] = pd.to_numeric(all_meals["total_TOTSUG"], errors="coerce")
    all_meals["_fat_val"] = pd.to_numeric(all_meals["total_FAT"], errors="coerce")
    all_meals["_prot_val"] = pd.to_numeric(all_meals["total_PROT"], errors="coerce")

    # Group all_meals by participant for fast lookups
    meals_by_pid = {}
    for pid, grp in all_meals.groupby("participant_id"):
        meals_by_pid[pid] = grp.sort_values("_ct").reset_index(drop=True)

    n_computed = 0

    for idx in df.index:
        pid = df.loc[idx, "participant_id"]
        ct_str = df.loc[idx, "corrected_time"]

        try:
            t0 = pd.Timestamp(ct_str)
        except Exception:
            continue

        # 5d. Hour of day (local)
        df.loc[idx, "hour_of_day"] = t0.hour

        # 5e. Meal label encoding
        ml = str(df.loc[idx, "meal_label"]).lower()
        df.loc[idx, "is_breakfast"] = 1 if "breakfast" in ml else 0
        df.loc[idx, "is_lunch"] = 1 if "lunch" in ml else 0
        df.loc[idx, "is_dinner"] = 1 if "evening dinner" in ml or "dinner" in ml else 0
        # Snack only if no main meal in the group
        has_main = ("breakfast" in ml or "lunch" in ml or "dinner" in ml)
        df.loc[idx, "is_snack"] = 1 if ("snack" in ml and not has_main) else 0

        # Need participant timeline for temporal features
        p_meals = meals_by_pid.get(pid)
        if p_meals is None or len(p_meals) == 0:
            n_computed += 1
            continue

        prior = p_meals[p_meals["_ct"] < t0]

        # 5a. Past 3h nutritional intake
        cutoff_3h = t0 - pd.Timedelta(hours=3)
        recent = prior[prior["_ct"] >= cutoff_3h]
        if len(recent) > 0:
            df.loc[idx, "past_3h_kcal"] = round(recent["_kcal_val"].sum(), 2)
            df.loc[idx, "past_3h_cho"] = round(recent["_cho_val"].sum(), 2)
            df.loc[idx, "past_3h_sugar"] = round(recent["_totsug_val"].sum(), 2)
            df.loc[idx, "past_3h_fat"] = round(recent["_fat_val"].sum(), 2)
            df.loc[idx, "past_3h_prot"] = round(recent["_prot_val"].sum(), 2)
        else:
            df.loc[idx, "past_3h_kcal"] = 0
            df.loc[idx, "past_3h_cho"] = 0
            df.loc[idx, "past_3h_sugar"] = 0
            df.loc[idx, "past_3h_fat"] = 0
            df.loc[idx, "past_3h_prot"] = 0

        # 5b. Time since last meal
        if len(prior) > 0:
            last_meal_time = prior["_ct"].iloc[-1]
            df.loc[idx, "time_since_last_meal_min"] = round(
                (t0 - last_meal_time).total_seconds() / 60, 1
            )

        # 5c. Time since last significant meal (CHO > 10g)
        sig_prior = prior[prior["_cho_val"] > 10]
        if len(sig_prior) > 0:
            last_sig_time = sig_prior["_ct"].iloc[-1]
            df.loc[idx, "time_since_last_sig_meal_min"] = round(
                (t0 - last_sig_time).total_seconds() / 60, 1
            )

        n_computed += 1

    print(f"    Computed temporal features for {n_computed} rows")
    for c in dt_cols:
        pct = 100 * df[c].notna().sum() / len(df) if len(df) > 0 else 0
        print(f"      {c:35s} {pct:5.1f}% non-null")

    return df


# ── Step 6: Derived Nutrient Ratios ─────────────────────────────
def step6_derived_ratios(df):
    """Compute derived nutrient ratio features."""
    print("\n  Step 6: Derived Nutrient Ratios (Dc enrichment)")
    print("  " + "-" * 40)

    cho = pd.to_numeric(df["CHO"], errors="coerce").fillna(0)
    totsug = pd.to_numeric(df["TOTSUG"], errors="coerce").fillna(0)
    star = pd.to_numeric(df["STAR"], errors="coerce").fillna(0)
    free_sug = pd.to_numeric(df["FREE_SUGAR"], errors="coerce").fillna(0)
    gluc = pd.to_numeric(df["GLUC"], errors="coerce").fillna(0)
    sucr = pd.to_numeric(df["SUCR"], errors="coerce").fillna(0)
    malt = pd.to_numeric(df["MALT"], errors="coerce").fillna(0)
    fat = pd.to_numeric(df["FAT"], errors="coerce").fillna(0)
    prot = pd.to_numeric(df["PROT"], errors="coerce").fillna(0)
    fibre = pd.to_numeric(df["AOACFIB"], errors="coerce").fillna(0)
    n6 = pd.to_numeric(df["TOTn6PFAC"], errors="coerce").fillna(0)
    n3 = pd.to_numeric(df["TOTn3PFAC"], errors="coerce").fillna(0)

    cho_safe = cho.clip(lower=0.1)
    totsug_safe = totsug.clip(lower=0.1)
    cho_safe_1 = cho.clip(lower=1.0)
    n3_safe = n3.clip(lower=0.01)

    df["starch_fraction"] = (star / cho_safe).round(4)
    df["sugar_fraction"] = (totsug / cho_safe).round(4)
    df["free_sugar_fraction"] = (free_sug / cho_safe).round(4)
    df["rapid_glucose_equiv"] = (gluc + sucr * 0.5 + malt).round(4)
    df["intrinsic_sugar"] = (totsug - free_sug).round(4)
    df["fat_cho_ratio"] = (fat / cho_safe).round(4)
    df["protein_cho_ratio"] = (prot / cho_safe).round(4)
    df["fibre_cho_ratio"] = (fibre / cho_safe).round(4)
    df["fat_sugar_ratio"] = (fat / totsug_safe).round(4)
    df["protein_sugar_ratio"] = (prot / totsug_safe).round(4)
    df["glycaemic_brake"] = (
        (prot * 3.27 + fat * 1.54 + fibre * 2.0) / cho_safe_1
    ).round(4)
    df["n6_n3_ratio"] = (n6 / n3_safe).round(4)

    ratio_cols = ["starch_fraction", "sugar_fraction", "free_sugar_fraction",
                  "rapid_glucose_equiv", "intrinsic_sugar",
                  "fat_cho_ratio", "protein_cho_ratio", "fibre_cho_ratio",
                  "fat_sugar_ratio", "protein_sugar_ratio",
                  "glycaemic_brake", "n6_n3_ratio"]

    print(f"    Computed {len(ratio_cols)} derived ratios")
    return df


# ── Step 7: Participant-Level Features ──────────────────────────
def step7_participant_features(df, all_meals, sex_map):
    """Compute per-participant features and merge."""
    print("\n  Step 7: Participant-Level Features")
    print("  " + "-" * 40)

    all_meals = all_meals.copy()
    all_meals["_date_norm"] = all_meals["date"].apply(_normalise_date)
    all_meals["_kcal"] = pd.to_numeric(all_meals["total_KCALS"], errors="coerce")
    all_meals["_cho"] = pd.to_numeric(all_meals["total_CHO"], errors="coerce")

    plevel = []
    for pid, grp in all_meals.groupby("participant_id"):
        n_meals = len(grp)
        dates = grp["_date_norm"].dropna().unique()
        n_days = len(dates)

        daily_kcal = grp.groupby("_date_norm")["_kcal"].sum()
        daily_cho = grp.groupby("_date_norm")["_cho"].sum()

        mfid = grp["myfood24_id"].iloc[0]
        sex = sex_map.get(mfid, np.nan)

        plevel.append({
            "participant_id": pid,
            "sex": sex,
            "n_total_meals": n_meals,
            "n_days_tracked": n_days,
            "mean_daily_kcal": round(daily_kcal.mean(), 1) if len(daily_kcal) > 0 else np.nan,
            "mean_daily_cho": round(daily_cho.mean(), 1) if len(daily_cho) > 0 else np.nan,
        })

    plevel_df = pd.DataFrame(plevel)
    print(f"    {len(plevel_df)} participants")

    # Merge onto df
    p_cols_to_merge = ["participant_id", "sex", "n_total_meals", "n_days_tracked",
                       "mean_daily_kcal", "mean_daily_cho"]
    df = df.merge(plevel_df[p_cols_to_merge], on="participant_id", how="left",
                  suffixes=("", "_plevel"))

    # Resolve any suffix collisions
    if "sex_plevel" in df.columns:
        df["sex"] = df["sex_plevel"].fillna(df.get("sex", np.nan))
        df.drop(columns=["sex_plevel"], inplace=True)

    for c in ["n_total_meals", "n_days_tracked", "mean_daily_kcal", "mean_daily_cho"]:
        if f"{c}_plevel" in df.columns:
            df[c] = df[f"{c}_plevel"]
            df.drop(columns=[f"{c}_plevel"], inplace=True)

    print(f"    Sex distribution:")
    sex_c = plevel_df["sex"].value_counts()
    for s, n in sex_c.items():
        print(f"      {s}: {n}")

    return df


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Feature Matrix Builder — XGBoost iAUC Prediction")
    print("=" * 72)

    # ── Load data ─────────────────────────────────────────────
    print("\n  Loading data...")
    meals = pd.read_csv(MEAL_FILE, low_memory=False)
    print(f"    Meal events: {len(meals)} rows")

    extract = pd.read_csv(EXTRACT_FILE, low_memory=False)
    print(f"    Patient extract: {len(extract)} rows")

    # Sex mapping from source file
    source = pd.read_csv(SOURCE_FILE, usecols=["Patient Id", "Sex"], low_memory=False)
    sex_map = source.drop_duplicates("Patient Id").set_index("Patient Id")["Sex"].to_dict()

    # CGM files
    cgm_files = {f.stem.replace("CGM_", ""): f for f in CGM_DIR.glob("CGM_*.csv")}
    print(f"    CGM files: {len(cgm_files)}")

    cgm_cache = {}

    # Keep a copy of all_meals for temporal lookups (pre-aggregation)
    all_meals = meals.copy()

    # ── Pipeline ──────────────────────────────────────────────
    meals = step1_nutrient_enrichment(meals, extract)
    df = step2_aggregate_stacking(meals)
    df, cgm_cache = step3_compute_iauc(df, cgm_cache, cgm_files)
    df = step4_glycaemic_features(df, cgm_cache)
    df = step5_diet_temporal(df, all_meals)
    df = step6_derived_ratios(df)
    df = step7_participant_features(df, all_meals, sex_map)

    # ── Assemble output columns ───────────────────────────────
    print("\n  Assembling output...")

    # Rename event_id to event_ids for clarity (may contain multiple)
    if "event_id" in df.columns:
        df.rename(columns={"event_id": "event_ids"}, inplace=True)

    # Rename columns for output (food_items -> food_items_concat)
    if "food_items" in df.columns:
        df.rename(columns={"food_items": "food_items_concat"}, inplace=True)

    # Rename meal_label -> meal_labels
    if "meal_label" in df.columns:
        df.rename(columns={"meal_label": "meal_labels"}, inplace=True)

    # Rename n_items -> n_food_items
    if "n_items" in df.columns:
        df.rename(columns={"n_items": "n_food_items"}, inplace=True)

    # Ordered column layout
    id_cols = ["participant_id", "excursion_id", "event_ids", "date",
               "meal_labels", "n_meals_in_excursion", "n_food_items",
               "food_items_concat"]

    target_cols = ["iAUC_mmol_min"]

    quality_cols = ["confidence", "match_type", "batch_day",
                    "baseline_glucose_mmol", "n_readings", "pct_coverage",
                    "max_gap_min", "iauc_status"]

    dc_nutrient_cols = [
        "totalVeg", "totalFruit", "WATER", "PROT", "FAT", "CHO", "KCALS", "KJ",
        "STAR", "OLIGO", "TOTSUG", "GLUC", "GALACT", "FRUCT", "SUCR", "MALT",
        "LACT", "ALCO", "ENGFIB", "AOACFIB",
        "SATFAC", "TOTn6PFAC", "TOTn3PFAC", "MONOFACc", "POLYFACc", "FACTRANS",
        "MG", "FE", "ZN", "MN", "SE", "VITD",
        "FREE_SUGAR", "ADDED_SUGAR", "CAFF",
    ]

    dc_ratio_cols = [
        "starch_fraction", "sugar_fraction", "free_sugar_fraction",
        "rapid_glucose_equiv", "intrinsic_sugar",
        "fat_cho_ratio", "protein_cho_ratio", "fibre_cho_ratio",
        "fat_sugar_ratio", "protein_sugar_ratio",
        "glycaemic_brake", "n6_n3_ratio",
    ]

    g_cols = [
        "baseline_glucose_mmol",
        "past_4h_glucose_trend",
        "past_1h_glucose_mean", "past_1h_glucose_sd", "past_1h_glucose_range",
        "mean_glucose_24h", "sd_glucose_24h", "cv_glucose_24h",
        "glucose_at_t_minus_15", "glucose_at_t_minus_30",
    ]

    dt_cols = [
        "past_3h_kcal", "past_3h_cho", "past_3h_sugar", "past_3h_fat", "past_3h_prot",
        "time_since_last_meal_min", "time_since_last_sig_meal_min",
        "hour_of_day",
        "is_breakfast", "is_lunch", "is_dinner", "is_snack",
    ]

    participant_cols = ["sex", "n_total_meals", "n_days_tracked",
                        "mean_daily_kcal", "mean_daily_cho"]

    validation_cols = ["excursion_rise_mmol", "excursion_peak_mmol",
                       "peak_glucose_mmol", "time_to_peak_min",
                       "glucose_at_120min_mmol", "iAUC_mmol_h"]

    # baseline_glucose_mmol appears in both quality and G — deduplicate
    all_ordered = []
    seen = set()
    for col in (id_cols + target_cols + quality_cols + dc_nutrient_cols +
                dc_ratio_cols + g_cols + dt_cols + participant_cols +
                validation_cols):
        if col not in seen and col in df.columns:
            all_ordered.append(col)
            seen.add(col)

    df_out = df[all_ordered].copy()

    # ── Filter: keep only rows with computed iAUC ─────────────
    n_before = len(df_out)
    df_out = df_out[df_out["iAUC_mmol_min"].notna()].reset_index(drop=True)
    print(f"\n  Filtered: {n_before} -> {len(df_out)} rows (kept only computed iAUC)")

    # ── Save ──────────────────────────────────────────────────
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE.name}")
    print(f"  {len(df_out)} rows, {len(df_out.columns)} columns")

    # ── Validation ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VALIDATION")
    print("=" * 72)

    # 1. iAUC summary
    computed = df_out[df_out["iAUC_mmol_min"].notna()]
    if len(computed) > 0:
        iauc = computed["iAUC_mmol_min"]
        print(f"\n  iAUC summary (n={len(computed)}):")
        print(f"    Mean:   {iauc.mean():.1f} mmol·min/L")
        print(f"    Median: {iauc.median():.1f} mmol·min/L")
        print(f"    SD:     {iauc.std():.1f}")
        print(f"    Range:  [{iauc.min():.1f}, {iauc.max():.1f}]")

    # 2. Benchmark: 40-60g CHO meals
    if "CHO" in df_out.columns:
        cho_vals = pd.to_numeric(df_out["CHO"], errors="coerce")
        mid_cho = computed[(cho_vals.loc[computed.index] >= 40) &
                           (cho_vals.loc[computed.index] <= 60)]
        if len(mid_cho) > 10:
            med = mid_cho["iAUC_mmol_min"].median()
            print(f"\n  Benchmark (~50g CHO, n={len(mid_cho)}):")
            print(f"    Median iAUC: {med:.1f} mmol·min/L")
            if 99 <= med <= 180:
                print(f"    -> Within expected range (99-180)")
            else:
                print(f"    -> Outside expected range (99-180)")

    # 3. Correlation check
    rise_pipe = pd.to_numeric(df_out["excursion_rise_mmol"], errors="coerce")
    peak_raw = pd.to_numeric(df_out["peak_glucose_mmol"], errors="coerce")
    bl_raw = pd.to_numeric(df_out["baseline_glucose_mmol"], errors="coerce")
    rise_raw = peak_raw - bl_raw
    valid = rise_pipe.notna() & rise_raw.notna()
    if valid.sum() > 10:
        corr = np.corrcoef(rise_pipe[valid], rise_raw[valid])[0, 1]
        print(f"\n  Correlation (pipeline_rise vs raw_rise): {corr:.3f} (n={valid.sum()})")

    # 4. Feature completeness
    print(f"\n  Feature completeness:")
    feature_groups = {
        "Dc nutrients": dc_nutrient_cols,
        "Dc ratios": dc_ratio_cols,
        "G glycaemic": [c for c in g_cols if c in df_out.columns],
        "Dt temporal": dt_cols,
        "Participant": participant_cols,
    }
    for gname, cols in feature_groups.items():
        cols_exist = [c for c in cols if c in df_out.columns]
        if cols_exist:
            pcts = [df_out[c].notna().mean() * 100 for c in cols_exist]
            print(f"    {gname:20s}  {min(pcts):5.1f}% - {max(pcts):5.1f}%  "
                  f"(avg {np.mean(pcts):.1f}%)")

    # 5. Row counts by status
    print(f"\n  Status distribution:")
    for s, n in df_out["iauc_status"].value_counts().items():
        print(f"    {s:30s} {n:>5d}  ({100*n/len(df_out):5.1f}%)")

    # 6. Participant count
    print(f"\n  Participants: {df_out['participant_id'].nunique()}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
