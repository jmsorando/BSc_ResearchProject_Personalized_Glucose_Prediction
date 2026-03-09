#!/usr/bin/env python3
"""
Generate an updated patient_extract CSV with CGM-realigned meal times.

Reads the original patient_extract0912_filtered_corrected.csv and the
pipeline output corrected_meal_times_ALL.csv, maps each food item row
to its meal bundle, and applies the time shift from the CGM realignment.

Output: patient_extract1602_realigned.csv
  - Same columns as source
  - "Time consumed at" replaced with the CGM-corrected time
  - New column "time_shift_min" showing the shift applied per row
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

BASE = Path(r"C:\Users\Jose Miguel Sorando\Documents\RP Cleaning 5")
SRC  = BASE / "source"
OUT  = BASE / "output"


def main():
    # ── Load data ─────────────────────────────────────────────────
    print("Loading source CSV...")
    source = pd.read_csv(SRC / "patient_extract1602.csv", low_memory=False)
    print(f"  {len(source)} rows, {len(source.columns)} columns")

    print("Loading corrected_meal_times_ALL.csv...")
    corrected = pd.read_csv(OUT / "corrected_meal_times_ALL.csv", low_memory=False)
    print(f"  {len(corrected)} meal bundles")

    # ── Parse source ──────────────────────────────────────────────
    source["Patient Id"] = source["Patient Id"].astype(str).str.strip()
    source["_date"] = pd.to_datetime(
        source["Date"], format="mixed", dayfirst=False
    ).dt.date

    def parse_time_to_dt(row):
        try:
            pts = str(row["Time consumed at"]).split(":")
            h = int(pts[0])
            m = int(pts[1]) if len(pts) > 1 else 0
            d = row["_date"]
            return datetime(d.year, d.month, d.day, h, m)
        except Exception:
            return pd.NaT

    source["_parsed_time"] = source.apply(parse_time_to_dt, axis=1)

    # ── Parse corrected bundles ───────────────────────────────────
    corrected["myfood24_id"] = corrected["myfood24_id"].astype(str).str.strip()
    corrected["date"] = pd.to_datetime(corrected["date"]).dt.date
    corrected["reported_time"] = pd.to_datetime(
        corrected["reported_time"], errors="coerce"
    )
    corrected["corrected_time"] = pd.to_datetime(
        corrected["corrected_time"], errors="coerce"
    )
    corrected["time_shift_min"] = (
        pd.to_numeric(corrected["time_shift_min"], errors="coerce").fillna(0.0)
    )

    # Build lookup by (myfood24_id, date)
    bundle_lookup = defaultdict(list)
    for _, row in corrected.iterrows():
        key = (str(row["myfood24_id"]), row["date"])
        foods = set(
            f.strip() for f in str(row["food_items"]).split(";") if f.strip()
        )
        bundle_lookup[key].append({
            "meal_label": str(row["meal_label"]),
            "food_items_set": foods,
            "food_items_str": str(row["food_items"]),
            "reported_time": row["reported_time"],
            "time_shift_min": row["time_shift_min"],
            "n_items": int(row["n_items"]) if pd.notna(row["n_items"]) else 1,
            "_matched_count": 0,
        })

    # ── Match source rows to bundles ──────────────────────────────
    print("Matching source rows to meal bundles...")
    shifts = np.zeros(len(source))
    matched_count = 0

    for i, row in source.iterrows():
        key = (str(row["Patient Id"]), row["_date"])
        bundles = bundle_lookup.get(key, [])
        if not bundles:
            continue

        food_name = str(row["Food name"]).strip()
        meal = str(row["Meal"]).strip()
        parsed_time = row["_parsed_time"]

        best_bundle = None
        best_score = -1.0

        for b in bundles:
            score = 0.0

            # Food name exact set match (strongest signal)
            if food_name in b["food_items_set"]:
                score += 100
            # Substring fallback (handles names with semicolons)
            elif food_name in b["food_items_str"]:
                score += 50

            # Meal label match
            if meal == b["meal_label"]:
                score += 20

            # Time proximity bonus (up to 30 points)
            if pd.notna(parsed_time) and pd.notna(b["reported_time"]):
                diff_min = abs(
                    (parsed_time - b["reported_time"]).total_seconds() / 60
                )
                score += max(0, 30 - diff_min / 5)

            # Penalise over-matched bundles
            if b["_matched_count"] >= b["n_items"]:
                score -= 50

            if score > best_score:
                best_score = score
                best_bundle = b

        if best_bundle is not None and best_score > 0:
            shifts[i] = best_bundle["time_shift_min"]
            best_bundle["_matched_count"] += 1
            matched_count += 1

    print(f"  Matched {matched_count}/{len(source)} rows")

    # ── Apply shifts ──────────────────────────────────────────────
    print("Applying time shifts...")
    new_times = []
    for i, row in source.iterrows():
        shift = shifts[i]
        parsed = row["_parsed_time"]
        if pd.notna(parsed) and shift != 0:
            corrected_dt = parsed + timedelta(minutes=shift)
            new_times.append(f"{corrected_dt.hour}:{corrected_dt.minute:02d}")
        else:
            new_times.append(row["Time consumed at"])

    source["Time consumed at"] = new_times
    source["time_shift_min"] = shifts

    # ── Clean up temp columns and save ────────────────────────────
    source.drop(columns=["_date", "_parsed_time"], inplace=True)

    out_path = OUT / "patient_extract1602_realigned.csv"
    source.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.name}")
    print(f"  {len(source)} rows, {len(source.columns)} columns")

    # ── Summary stats ─────────────────────────────────────────────
    shifted = (shifts != 0).sum()
    print(f"  {shifted} rows had time shifts applied")
    if shifted > 0:
        nonzero = shifts[shifts != 0]
        print(f"  Mean shift:   {nonzero.mean():.1f} min")
        print(f"  Median shift: {np.median(nonzero):.1f} min")
        print(f"  Range: [{nonzero.min():.1f}, {nonzero.max():.1f}] min")


if __name__ == "__main__":
    main()
