#!/usr/bin/env python3
"""
Compute incremental Area Under the Curve (iAUC) of glucose for 2-hour
postprandial windows using aligned meal data and raw CGM traces.

iAUC uses the positive-only trapezoidal rule (Wolever/FAO convention):
only glucose increments above the pre-meal baseline (nadir) are summed.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import Counter

# ── Config ────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\Jose Miguel Sorando\Documents\RP Cleaning 4 (Claude)")
CGM_DIR     = BASE / "source" / "cgm_data"
MEAL_FILE   = BASE / "output" / "corrected_meal_times_ALL.csv"
OUTPUT_FILE = BASE / "output" / "iauc_meal_events.csv"

WINDOW_MIN        = 120   # 2-hour postprandial window
EXPECTED_READINGS = 25    # 120 min / 5 min + 1
MAX_NADIR_SNAP    = 10    # max minutes between nadir and nearest CGM reading
GAP_FLAG_MIN      = 15    # flag events with gaps exceeding this


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

    # Handle sentinel strings
    cgm["glucose"] = cgm["glucose"].replace({"Low": 2.2, "High": 22.2})
    cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")

    cgm = cgm.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    return cgm[["ts_utc", "glucose"]].copy()


def _compute_iauc(cgm_window, g0):
    """
    Trapezoidal iAUC (positive-only) over a CGM window.

    Parameters
    ----------
    cgm_window : DataFrame with ts_utc and glucose, sorted by time
    g0         : baseline glucose (mmol/L)

    Returns
    -------
    iauc : float (mmol·min/L)
    """
    ts = cgm_window["ts_utc"].values
    gl = cgm_window["glucose"].values
    iauc = 0.0

    for i in range(len(gl) - 1):
        if np.isnan(gl[i]) or np.isnan(gl[i + 1]):
            continue
        d0 = max(gl[i] - g0, 0.0)
        d1 = max(gl[i + 1] - g0, 0.0)
        dt = (ts[i + 1] - ts[i]) / np.timedelta64(1, "m")  # minutes
        iauc += 0.5 * (d0 + d1) * dt

    return iauc


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  iAUC Computation from CGM + Aligned Meal Data")
    print("=" * 72)

    # Load meal events
    print(f"\n  Loading {MEAL_FILE.name}...")
    meals = pd.read_csv(MEAL_FILE, low_memory=False)
    print(f"  {len(meals)} meal events")

    meals["nadir_time"] = pd.to_datetime(meals["nadir_time"], errors="coerce")
    meals["peak_time"] = pd.to_datetime(meals["peak_time"], errors="coerce")

    # Discover CGM files
    cgm_files = {f.stem.replace("CGM_", ""): f for f in CGM_DIR.glob("CGM_*.csv")}
    print(f"  {len(cgm_files)} CGM files found")

    # Prepare output columns
    out_cols = ["baseline_glucose_mmol", "iAUC_mmol_min", "iAUC_mmol_h",
                "peak_glucose_mmol", "time_to_peak_min", "glucose_at_120min_mmol",
                "n_readings", "pct_coverage", "max_gap_min", "iauc_status"]
    for c in out_cols:
        meals[c] = np.nan
    meals["iauc_status"] = ""

    # Cache CGM data per participant
    cgm_cache = {}
    status_counts = Counter()

    pids = meals["participant_id"].unique()
    print(f"\n  Processing {len(pids)} participants...")

    for pi, pid in enumerate(sorted(pids)):
        sub = meals[meals["participant_id"] == pid]
        n_with_nadir = sub["nadir_time"].notna().sum()

        if n_with_nadir == 0:
            # No matched events for this participant — set status from confidence
            for idx in sub.index:
                conf = str(meals.loc[idx, "confidence"])
                meals.loc[idx, "iauc_status"] = conf
                status_counts[conf] += 1
            continue

        # Load CGM (with caching)
        if pid not in cgm_cache:
            if pid in cgm_files:
                cgm_cache[pid] = _load_cgm(cgm_files[pid])
            else:
                cgm_cache[pid] = None

        cgm = cgm_cache[pid]

        for idx in sub.index:
            nadir = meals.loc[idx, "nadir_time"]

            # Events without nadir — use their confidence as status
            if pd.isna(nadir):
                conf = str(meals.loc[idx, "confidence"])
                meals.loc[idx, "iauc_status"] = conf
                status_counts[conf] += 1
                continue

            # No CGM file
            if cgm is None or len(cgm) == 0:
                meals.loc[idx, "iauc_status"] = "no_cgm_file"
                status_counts["no_cgm_file"] += 1
                continue

            # Convert nadir from local to UTC
            tz_off = _parse_tz_offset(meals.loc[idx, "tz_offset"])
            nadir_utc = pd.Timestamp(nadir).tz_localize(None) - tz_off
            nadir_utc = nadir_utc.tz_localize("UTC")

            # Find nearest CGM reading to nadir
            diffs = (cgm["ts_utc"] - nadir_utc).abs()
            nearest_idx = diffs.idxmin()
            snap_min = diffs.loc[nearest_idx].total_seconds() / 60

            if snap_min > MAX_NADIR_SNAP:
                meals.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue

            g0 = cgm.loc[nearest_idx, "glucose"]
            if np.isnan(g0):
                meals.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue

            # Extract 2h window [nadir_utc, nadir_utc + 120 min]
            window_end = nadir_utc + pd.Timedelta(minutes=WINDOW_MIN)
            mask = (cgm["ts_utc"] >= nadir_utc - pd.Timedelta(seconds=30)) & \
                   (cgm["ts_utc"] <= window_end + pd.Timedelta(seconds=30))
            window = cgm.loc[mask].copy()

            if len(window) < 3:
                meals.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue

            # Quality metrics
            n_readings = len(window)
            pct_cov = n_readings / EXPECTED_READINGS
            gaps = window["ts_utc"].diff().dt.total_seconds() / 60
            max_gap = gaps.max() if len(gaps) > 1 else 0.0

            # Compute iAUC
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

            # Glucose at 120 min (nearest reading to window end)
            end_diffs = (window["ts_utc"] - window_end).abs()
            end_idx = end_diffs.idxmin()
            end_snap = end_diffs.loc[end_idx].total_seconds() / 60
            gl_120 = window.loc[end_idx, "glucose"] if end_snap <= 10 else np.nan

            # Status
            if max_gap > GAP_FLAG_MIN:
                status = "gap_too_large"
            else:
                status = "ok"

            meals.loc[idx, "baseline_glucose_mmol"] = round(g0, 4)
            meals.loc[idx, "iAUC_mmol_min"] = round(iauc, 4)
            meals.loc[idx, "iAUC_mmol_h"] = round(iauc / 60, 4)
            meals.loc[idx, "peak_glucose_mmol"] = round(peak_gl, 4)
            meals.loc[idx, "time_to_peak_min"] = round(ttp, 1)
            meals.loc[idx, "glucose_at_120min_mmol"] = round(gl_120, 4) if not np.isnan(gl_120) else np.nan
            meals.loc[idx, "n_readings"] = int(n_readings)
            meals.loc[idx, "pct_coverage"] = round(pct_cov, 4)
            meals.loc[idx, "max_gap_min"] = round(max_gap, 1)
            meals.loc[idx, "iauc_status"] = status
            status_counts[status] += 1

        if (pi + 1) % 10 == 0 or pi + 1 == len(pids):
            print(f"    [{pi+1}/{len(pids)}] {pid}")

    # ── Reorder columns: put iAUC block right after meal_label ────
    iauc_cols = ["baseline_glucose_mmol", "iAUC_mmol_min", "iAUC_mmol_h",
                 "peak_glucose_mmol", "time_to_peak_min", "glucose_at_120min_mmol",
                 "n_readings", "pct_coverage", "max_gap_min", "iauc_status"]
    all_cols = list(meals.columns)
    anchor = "meal_label"
    if anchor in all_cols:
        pos = all_cols.index(anchor) + 1
        remaining = [c for c in all_cols if c not in iauc_cols]
        ordered = remaining[:pos] + iauc_cols + remaining[pos:]
        meals = meals[ordered]

    # ── Save output ───────────────────────────────────────────────
    meals.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved: {OUTPUT_FILE.name}  ({len(meals)} rows)")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  iAUC SUMMARY")
    print("=" * 72)

    print("\n  Status counts:")
    for s, n in status_counts.most_common():
        print(f"    {s:35s} {n:>5d}  ({100*n/len(meals):5.1f}%)")

    ok = meals[meals["iauc_status"] == "ok"]
    gap_large = meals[meals["iauc_status"] == "gap_too_large"]
    computed = pd.concat([ok, gap_large]) if len(gap_large) > 0 else ok

    if len(computed) > 0:
        iauc_vals = computed["iAUC_mmol_min"].dropna()
        bl_vals = computed["baseline_glucose_mmol"].dropna()
        cov_vals = computed["pct_coverage"].dropna()

        print(f"\n  Computed iAUC events: {len(computed)}")
        print(f"    iAUC (mmol·min/L):")
        print(f"      Mean:   {iauc_vals.mean():.1f}")
        print(f"      Median: {iauc_vals.median():.1f}")
        print(f"      SD:     {iauc_vals.std():.1f}")
        print(f"      Range:  [{iauc_vals.min():.1f}, {iauc_vals.max():.1f}]")
        print(f"    Baseline glucose: {bl_vals.mean():.2f} mmol/L (mean)")
        print(f"    Coverage: {cov_vals.mean():.1%} (mean)")

        # Validation: correlation with pipeline's excursion_rise
        rise_pipe = computed["excursion_rise_mmol"].dropna()
        rise_iauc = computed.loc[rise_pipe.index, "peak_glucose_mmol"] - \
                    computed.loc[rise_pipe.index, "baseline_glucose_mmol"]
        valid = rise_pipe.notna() & rise_iauc.notna()
        if valid.sum() > 10:
            corr = np.corrcoef(rise_pipe[valid], rise_iauc[valid])[0, 1]
            print(f"\n  Validation:")
            print(f"    Corr(pipeline_rise, raw_rise): {corr:.3f}  (n={valid.sum()})")

        # Benchmark check for ~50g CHO meals
        cho_col = "total_CHO"
        if cho_col in computed.columns:
            mid_cho = computed[(computed[cho_col] >= 40) & (computed[cho_col] <= 60)]
            if len(mid_cho) > 10:
                med = mid_cho["iAUC_mmol_min"].median()
                print(f"    ~50g CHO meals (n={len(mid_cho)}): median iAUC = {med:.1f} mmol·min/L")
                if 99 <= med <= 180:
                    print(f"    -> Within expected range (99-180)")
                else:
                    print(f"    -> Outside expected range (99-180) — review baseline/units")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
