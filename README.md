# CGM-First Meal Time Re-Alignment & Feature Matrix Pipeline

A three-stage Python pipeline that (1) aligns self-reported food diary meal times with continuous glucose monitoring (CGM) data, (2) produces a realigned food diary, and (3) builds an XGBoost-ready feature matrix with iAUC targets for postprandial glucose prediction.

---

## Philosophy

**The CGM is ground truth for WHEN eating happened. The food diary tells us WHAT was eaten. The pipeline assigns WHAT to WHEN.**

We detect eating events from glucose excursions first, then assign food diary entries to those events. The CGM drives the timeline; the diary provides nutritional content.

---

## Data Quality Landscape

Analysis of the full dataset (993 participant-days, 105 participant-IDs) reveals four scenarios:

| Scenario | Frequency | Description |
|---|---|---|
| **24h + real-time** | ~71% | Logged throughout the day in 24h format. Times approximately correct. |
| **24h + batch** | ~24% | Logged entire day in 1-2 sessions. Ordering usually correct, absolute times may be off. |
| **12h + real-time** | ~5% | Logged throughout the day in 12h format without AM/PM. Times ambiguous. |
| **12h + batch** | ~1% | Worst case: batch-logged AND 12h ambiguity. |

9 out of 105 participants use 12h format. 237 out of 993 participant-days are batch-entry days. The pipeline handles all four scenarios.

---

## Pipeline Overview

The pipeline runs in three stages, each a separate script:

```
Stage 1                          Stage 2                          Stage 3
cgm_meal_realignment.py    -->   generate_realigned_source.py --> build_feature_matrix.py

Inputs:                          Inputs:                          Inputs:
  patient_extract1602.csv          patient_extract1602.csv          corrected_meal_times_ALL.csv
  MyFood24 ID Matched.csv         corrected_meal_times_ALL.csv     patient_extract1602_realigned.csv
  CGM_*.csv files                                                   CGM_*.csv files
                                                                    patient_extract1602.csv (sex)
Outputs:                         Output:                          Output:
  corrected_meal_times_ALL.csv     patient_extract1602_realigned    feature_matrix.csv
  processing_report.csv              .csv                           (2,228 rows, 96 columns)
  plots/ (75 per-participant
    + global_summary.png)
```

---

## Stage 1: CGM Meal Realignment (`cgm_meal_realignment.py`)

The main pipeline runs in 10 steps:

1. **Load ID mapping** -- reads participant-to-MyFood24 mapping, discovers CGM files
2. **Classify data quality** -- detects 12h vs 24h time format per participant; classifies each day as batch or real-time entry
3. **Process CGM** -- converts UTC to local time, interpolates short gaps (<=30 min), applies Savitzky-Golay smoothing, computes dG/dt
4. **Detect excursions** -- two-stage detection with steep-rise confirmation and nadir re-anchoring to avoid false triggers on flat/noisy traces
5. **Bundle diary entries** -- groups food items into meal events by meal label and entry timestamp proximity; cross-label temporal merging absorbs small snacks into adjacent meals
6. **Resolve AM/PM ambiguity** -- for 12h-format participants, uses CGM excursion scoring first, then `Item added at` timestamps, then meal-label heuristics
7. **Validate batch days** -- order-preserving assignment for batch-entry days (recalled order maps to excursion order)
8. **Match meals to excursions** -- three-pass approach:
   - **Pass 1 (Anchor)**: best meal per excursion via cost function (time distance + TOTSUG/rise similarity)
   - **Pass 2 (Stacking)**: remaining meals stack onto anchors if CGM confirms glucose still elevated
   - **Pass 3 (Conservative correction)**: high/medium shifts applied as-is; low-confidence shifts clamped to +/-30 min unless AM/PM or batch-day exceptions apply
9. **Assign confidence scores** -- each meal event gets a confidence tier (see table below)
10. **Produce outputs** -- writes corrected CSV, processing report, per-participant plots, and global summary plot

### Excursion Detection (Step 4 detail)

The two-stage approach prevents false triggers on flat-but-noisy CGM traces:

- **Stage 1**: Scan for sustained positive dG/dt (>0.02 mmol/L per 5 min for >=15 min). Filter candidates where the actual glucose rise within the window is <0.3 mmol/L (noise).
- **Stage 2**: For surviving candidates, find the first "steep rise" reading (dG/dt >0.08). Re-anchor the nadir by searching backward from the steep rise onset (not from the initial above-threshold reading). This correctly places the pre-meal nadir even when noisy readings triggered detection early.

### Confidence Tiers

| Confidence | Meaning | Shift Range |
|---|---|---|
| `high` | Strong CGM-diary agreement | < 30 min |
| `medium` | Moderate shift, well-supported | 30--90 min |
| `low_clamped` | Large shift on real-time day, capped to +/-30 min | 90--180 min |
| `low_batch_override` | Batch day -- large shift accepted (recalled times unreliable) | 90--180 min |
| `low_ampm_override` | 12h format -- CGM confirmed AM/PM fix | 90--180 min |
| `stacked` | Meal stacked onto existing excursion (glucose still elevated) | N/A |
| `rejected_implausible_shift` | Shift too large, reverted to reported time | > 180 min |
| `no_match` | Significant CHO but no suitable excursion | N/A |
| `low_cho_no_match` | CHO <= 5g (water, tea, black coffee) -- no match attempted | N/A |
| `no_cgm_data` | No CGM data for that day | N/A |
| `flat_trace` | CGM too flat (<2 excursions for >=3 significant meals) | N/A |
| `cgm_gap` | Meal falls in CGM dropout period (>30 min gap) | N/A |

### Confidence hierarchy (most to least trustworthy):
```
high > medium > stacked > low_ampm_override > low_batch_override > low_clamped > no_match > rejected
```

---

## Stage 2: Realigned Source Generation (`generate_realigned_source.py`)

Takes the original food diary (`patient_extract1602.csv`) and the pipeline output (`corrected_meal_times_ALL.csv`), maps each food item row to its meal bundle, and applies the time shift from the CGM realignment.

**Output**: `patient_extract1602_realigned.csv` -- same columns as source, with:
- `Time consumed at` replaced with CGM-corrected time
- New column `time_shift_min` showing the shift applied per row

---

## Stage 3: Feature Matrix Builder (`build_feature_matrix.py`)

Produces a single CSV (`feature_matrix.csv`) ready for XGBoost training. Each row is one glucose excursion (the unit of prediction). The target is the 2h postprandial iAUC. Only rows with a computed iAUC are included.

### Pipeline Steps

1. **Nutrient enrichment** -- joins patient_extract items back to meal events to recover the full nutrient panel (~35 columns). Disambiguates duplicate food names by time proximity.
2. **Excursion-level aggregation** -- groups stacked meals by `(participant_id, excursion_id)`, sums all nutrient columns. The iAUC reflects the combined nutritional input.
3. **iAUC computation** -- 2h postprandial incremental area under the curve (Wolever/FAO positive-only trapezoidal rule) from CGM readings anchored at the excursion nadir.
4. **Glycaemic features (G)** -- baseline glucose, 4h trend (OLS slope), 1h statistics (mean, SD, range), 24h metrics (mean, SD, CV), glucose at t-15 and t-30 min.
5. **Diet temporal features (Dt)** -- past 3h nutritional intake, time since last meal, time since last significant meal (CHO>10g), hour of day, meal label encoding.
6. **Derived nutrient ratios (Dc)** -- starch/sugar fractions, fat/CHO ratio, protein/CHO ratio, fibre/CHO ratio, glycaemic brake index, n6/n3 ratio, rapid glucose equivalent.
7. **Participant-level features** -- sex, total meals, days tracked, mean daily kcal/CHO.

### iAUC Computation Detail

```
1. Convert nadir_time (local) to UTC using tz_offset
2. Snap to nearest CGM reading (reject if >10 min away)
3. G0 = baseline glucose at nadir
4. Extract CGM window: [nadir, nadir + 120 min]
5. Trapezoidal integration (positive-only):
     For each consecutive pair (Gi, Gi+1):
       delta_i   = max(Gi - G0, 0)
       delta_i+1 = max(Gi+1 - G0, 0)
       iAUC += 0.5 * (delta_i + delta_i+1) * dt_minutes
6. Units: mmol*min/L
```

### Feature Matrix Columns (96 total)

```
IDENTIFIERS (8)
  participant_id, excursion_id, event_ids, date,
  meal_labels, n_meals_in_excursion, n_food_items, food_items_concat

TARGET (1)
  iAUC_mmol_min

QUALITY / FILTERING (8)
  confidence, match_type, batch_day,
  baseline_glucose_mmol, n_readings, pct_coverage, max_gap_min, iauc_status

DIET COMPOSITION -- Dc: Raw nutrients (35)
  CHO, STAR, TOTSUG, FREE_SUGAR, ADDED_SUGAR,
  GLUC, FRUCT, SUCR, MALT, LACT, GALACT, OLIGO,
  PROT, FAT, KCALS, KJ, ALCO, WATER,
  AOACFIB, ENGFIB,
  SATFAC, MONOFACc, POLYFACc, TOTn3PFAC, TOTn6PFAC, FACTRANS,
  MG, ZN, MN, FE, SE, VITD, CAFF,
  totalVeg, totalFruit

DIET COMPOSITION -- Dc: Derived ratios (12)
  starch_fraction, sugar_fraction, free_sugar_fraction,
  rapid_glucose_equiv, intrinsic_sugar,
  fat_cho_ratio, protein_cho_ratio, fibre_cho_ratio,
  fat_sugar_ratio, protein_sugar_ratio,
  glycaemic_brake, n6_n3_ratio

GLYCAEMIC CONTEXT -- G (10)
  baseline_glucose_mmol,
  past_4h_glucose_trend,
  past_1h_glucose_mean, past_1h_glucose_sd, past_1h_glucose_range,
  mean_glucose_24h, sd_glucose_24h, cv_glucose_24h,
  glucose_at_t_minus_15, glucose_at_t_minus_30

DIET TEMPORAL CONTEXT -- Dt (12)
  past_3h_kcal, past_3h_cho, past_3h_sugar, past_3h_fat, past_3h_prot,
  time_since_last_meal_min, time_since_last_sig_meal_min,
  hour_of_day,
  is_breakfast, is_lunch, is_dinner, is_snack

PARTICIPANT-LEVEL (5)
  sex, n_total_meals, n_days_tracked,
  mean_daily_kcal, mean_daily_cho

VALIDATION (6)
  excursion_rise_mmol, excursion_peak_mmol,
  peak_glucose_mmol, time_to_peak_min, glucose_at_120min_mmol,
  iAUC_mmol_h
```

---

## Project Structure

```
RP Cleaning 5/
├── source/                                    # Input data (do not modify)
│   ├── cgm_data/
│   │   └── CGM_<ParticipantID>.csv            # 95 CGM files
│   ├── patient_extract1602.csv                # Food diary (16,216 rows, 168 columns)
│   └── MyFood24 ID Matched(Sheet1).csv        # ID mapping (129 rows)
├── output/                                    # Generated results
│   ├── plots/
│   │   ├── <PID>_overview.png                 # 75 per-participant plots
│   │   └── global_summary.png
│   ├── corrected_meal_times_ALL.csv           # 5,200 meal events with corrections
│   ├── patient_extract1602_realigned.csv      # Realigned food diary (16,216 rows)
│   ├── processing_report.csv                  # Per-participant summary (105 rows)
│   └── feature_matrix.csv                     # XGBoost-ready matrix (2,228 rows, 96 cols)
├── cgm_meal_realignment.py                    # Stage 1: CGM meal realignment (10 steps)
├── generate_realigned_source.py               # Stage 2: Realigned source diary
├── build_feature_matrix.py                    # Stage 3: Feature matrix + iAUC
└── README.md
```

---

## Input Files

| File | Location | Description |
|---|---|---|
| `patient_extract1602.csv` | `source/` | MyFood24 food diary export. Columns: Patient Id, Sex, Date (M/D/YYYY), Time consumed at, Item added at, Meal, Food name, CHO, FAT, PROT, KCALS, TOTSUG, AOACFIB, + ~150 nutrient columns |
| `MyFood24 ID Matched(Sheet1).csv` | `source/` | Maps Participant ID (CGM ID) to MyFood24 ID. May contain compound IDs (e.g. `F105/T189`) or trailing spaces |
| `CGM_<ParticipantID>.csv` | `source/cgm_data/` | Per-participant CGM files (Dexcom). Columns: isoDate (ISO 8601 UTC), event_type, event_subtype, glucose (mmol/L, may contain "Low"/"High"), duration. 5-minute sampling intervals |

---

## Output Files

| File | Location | Description |
|---|---|---|
| `corrected_meal_times_ALL.csv` | `output/` | 5,200 meal events with original and corrected times, time shift, confidence, excursion details, 135 nutrient columns |
| `processing_report.csv` | `output/` | Per-participant summary (105 rows): match counts by confidence tier, excursion counts, mean/median shifts |
| `patient_extract1602_realigned.csv` | `output/` | Copy of source diary with corrected times and `time_shift_min` column (16,216 rows) |
| `feature_matrix.csv` | `output/` | XGBoost-ready matrix filtered to rows with computed iAUC (2,228 rows, 96 columns) |
| `plots/<PID>_overview.png` | `output/plots/` | Per-participant CGM overlay plots |
| `plots/global_summary.png` | `output/plots/` | Aggregate statistics: shift histogram, CHO vs rise scatter, confidence breakdown, shift by meal type |

---

## Per-Participant Plot Legend

Each subplot covers one day:

- **Steelblue line** -- raw CGM glucose trace
- **Dashed blue vertical line** -- reported meal time (label: B=Breakfast, L=Lunch, D=Dinner, S=Snack, Dr=Drink)
- **Solid orange vertical line** -- corrected meal time
- **Orange arrow** -- direction and magnitude of the time shift (label shows shift in minutes)
- **Green triangle** -- glucose nadir (start of excursion)
- **Yellow banner** -- batch-entry day indicator

---

## Configuration

### `cgm_meal_realignment.py`

| Parameter | Default | Description |
|---|---|---|
| `SMOOTHING_WINDOW` | 5 | Savitzky-Golay window size (odd) |
| `SMOOTHING_POLY` | 2 | Savitzky-Golay polynomial order |
| `PHYSIOLOGICAL_LAG_MIN` | 20 | Expected delay (min) from eating to glucose rise |
| `MIN_EXCURSION_RISE_MMOL` | 0.8 | Minimum nadir-to-peak rise to qualify as excursion |
| `DGDT_THRESHOLD` | 0.02 | dG/dt threshold for sustained rise detection (mmol/L per 5 min) |
| `STEEP_RISE_DGDT` | 0.08 | dG/dt threshold for unambiguous rise confirmation |
| `MIN_RISE_IN_WINDOW_MMOL` | 0.3 | Minimum glucose rise within sustained-rise window (noise filter) |
| `NADIR_REANCHOR_LOOKBACK` | 15 min | Lookback from steep rise onset for nadir re-anchoring |
| `EXCURSION_MERGE_MIN` | 25 | Merge excursions closer than this (minutes) |
| `MAX_ALLOWABLE_SHIFT_MIN` | 180 | Hard ceiling -- reject any match exceeding this |
| `LOW_CONFIDENCE_CAP_MIN` | 30 | Clamp applied to low-confidence shifts |
| `CHO_THRESHOLD` | 5 | Minimum carbohydrate (g) to attempt CGM matching |
| `BATCH_ENTRY_GAP_MIN` | 30 | Timestamp gap defining separate entry sessions |
| `STACKING_WINDOW_MIN` | 45 | Window for stacking a meal onto an existing excursion |
| `STACKING_ACTIVE_BUFFER_MIN` | 30 | Extra minutes past peak to extend active window |
| `AMPM_CGM_SEARCH_WINDOW_MIN` | 45 | Search radius for AM/PM CGM excursion scoring |

### `build_feature_matrix.py`

| Parameter | Default | Description |
|---|---|---|
| `WINDOW_MIN` | 120 | Postprandial iAUC window (minutes) |
| `MAX_NADIR_SNAP` | 10 | Max minutes between nadir time and nearest CGM reading |
| `GAP_FLAG_MIN` | 15 | Flag events with CGM gaps exceeding this (minutes) |
| `SENTINEL_LOW` | 2.2 | mmol/L replacement for "Low" CGM strings |
| `SENTINEL_HIGH` | 22.2 | mmol/L replacement for "High" CGM strings |

---

## Latest Run Results

| Metric | Value |
|---|---|
| Total meal events (Stage 1) | 5,200 |
| Participants | 105 (66F, 39M) |
| Participants with CGM match | 68 |
| Batch-entry days | 237 / 993 |
| 12h-format participants | 9 |
| Anchor matches | 2,232 |
| Stacked meals | 32 |
| Feature matrix rows (with iAUC) | 2,228 |
| Median iAUC | 114.0 mmol*min/L |
| Benchmark iAUC (40-60g CHO) | 116.5 mmol*min/L (within 99-180 range) |
| Pipeline rise vs raw rise correlation | r = 0.871 |
| Nutrient join success | 100% |
| Glycaemic feature completeness | ~99% (filtered dataset) |

---

## Downstream ML Usage

When training an XGBoost model on `feature_matrix.csv`:

- **Target**: `iAUC_mmol_min`
- **Features**: all Dc, G, and Dt columns (47 raw nutrients + 12 ratios + 10 glycaemic + 12 temporal + 5 participant = 86 features)
- **Filtering**: use `iauc_status == "ok"` for cleanest data (2,215 rows); include `gap_too_large` (13 rows) if coverage is acceptable
- **Stacked excursions**: nutrients are already aggregated across all meals sharing an excursion_id. The iAUC reflects the combined nutritional input.

---

## Requirements

- Python 3.10+
- pandas
- numpy
- scipy
- matplotlib

```bash
pip install pandas numpy scipy matplotlib
```

---

## Usage

```bash
# 1. Ensure source data is in source/ and source/cgm_data/
# 2. Run the three stages in order:

python cgm_meal_realignment.py          # Stage 1: ~2 min
python generate_realigned_source.py     # Stage 2: ~10 sec
python build_feature_matrix.py          # Stage 3: ~3 min

# All outputs are written to output/
```

---

## Edge Cases Handled

- **Shift workers**: No waking-hours filter. Excursions detected 24/7. Meal labels treated as descriptors, not clock constraints.
- **Overnight eating**: Cross-midnight meals search both current and previous day's excursions.
- **Compound participant IDs**: e.g. `F105/T189` -- split and processed separately.
- **Multiple MyFood24 IDs per participant**: e.g. B101 -> 2339, 2340 -- both processed.
- **Flat CGM traces**: Non-diabetic participants with <2 excursions for >=3 meals -> `flat_trace`.
- **CGM dropouts**: Gaps >30 min not interpolated; meals in gap periods -> `cgm_gap`.
- **Mislabelled meals**: Cross-label merging and stacking handle desserts logged as separate meals.
- **Mixed timezone offsets**: Most common offset per participant extracted from `Item added at`.

---

## License

This project is provided for research purposes.
