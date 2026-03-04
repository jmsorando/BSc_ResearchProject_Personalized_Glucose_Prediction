# CGM-First Meal Time Re-Alignment Pipeline

A Python pipeline that aligns self-reported food diary meal times with continuous glucose monitoring (CGM) data by detecting physiological glucose excursions and matching them to reported meals.

## Problem

Participants in dietary studies record meal times in food diaries (e.g. MyFood24), but these times are often inaccurate due to:

- **Batch entry** -- logging all meals at once at the end of the day
- **12/24-hour format ambiguity** -- unclear whether "1:00" means 01:00 or 13:00
- **Recall error** -- simply misremembering when a meal was eaten

This pipeline uses CGM glucose traces as ground truth to correct reported meal times by identifying when glucose actually rose in response to food intake.

## How It Works

The pipeline runs in 10 steps:

1. **Load data** -- reads the participant ID mapping, food diary CSV, and per-participant CGM files
2. **Classify data quality** -- detects 12h vs 24h time format; classifies each participant-day as batch or real-time entry
3. **Process CGM** -- converts UTC timestamps to local time, interpolates short gaps, applies Savitzky-Golay smoothing, and computes the glucose rate of change (dG/dt)
4. **Detect excursions** -- identifies sustained glucose rises (candidate meal responses) using dG/dt thresholds, anchors each to a nadir, and merges nearby excursions
5. **Bundle diary entries** -- groups food items into meal events by meal label and entry timestamp proximity
6. **Resolve AM/PM ambiguity** -- for 12h-format participants, uses CGM excursion scoring, `Item added at` timestamps, and meal label heuristics to disambiguate
7. *(reserved)*
8. **Match meals to excursions** -- assigns meal bundles to detected excursions using a cost function (time distance + sugar/rise similarity); handles batch days with rank-order matching and real-time days with greedy assignment. The cost function uses total sugars (TOTSUG) rather than total CHO for magnitude matching, since simple sugars drive the acute spikes detected by the algorithm. Total CHO (including starch and fibre) is still used for the threshold filter
9. *(reserved)*
10. **Produce outputs** -- writes corrected CSV, processing report, per-participant CGM plots, and a global summary plot

## iAUC Computation (`compute_iauc.py`)

After the main pipeline produces `corrected_meal_times_ALL.csv`, this script computes the **incremental Area Under the Curve (iAUC)** of glucose for each meal event using raw CGM traces. iAUC quantifies the total postprandial glucose exposure above baseline over a 2-hour window -- the standard metric for glycaemic response (Wolever/FAO convention).

### Method

1. **Baseline identification** -- for each meal event with a matched excursion, the nadir time (pre-meal glucose minimum) is converted from local time to UTC and snapped to the nearest CGM reading (within 10 min)
2. **Window extraction** -- a 2-hour CGM window is extracted from nadir to nadir + 120 min
3. **Trapezoidal iAUC** -- the positive-only trapezoidal rule is applied: only glucose increments above the baseline (nadir glucose) contribute to the area. Dips below baseline are clamped to zero
4. **Quality gating** -- events are flagged if CGM coverage is insufficient (< 3 readings) or if gaps exceed 15 min

### Output Columns (in `iauc_meal_events.csv`)

| Column | Description |
|--------|-------------|
| `baseline_glucose_mmol` | Glucose at the nadir (start of excursion), mmol/L |
| `iAUC_mmol_min` | Incremental area under the curve, mmol·min/L |
| `iAUC_mmol_h` | Same as above divided by 60, mmol·h/L |
| `peak_glucose_mmol` | Highest glucose in the 2h window, mmol/L |
| `time_to_peak_min` | Minutes from nadir to peak |
| `glucose_at_120min_mmol` | Glucose at end of 2h window, mmol/L |
| `n_readings` | Number of CGM readings in the window |
| `pct_coverage` | Proportion of expected readings present (25 expected for 2h at 5-min intervals) |
| `max_gap_min` | Largest gap between consecutive readings, minutes |
| `iauc_status` | `ok`, `gap_too_large`, `insufficient_cgm`, `no_cgm_file`, or passthrough confidence label |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WINDOW_MIN` | 120 | Postprandial window duration (minutes) |
| `MAX_NADIR_SNAP` | 10 | Max minutes between nadir time and nearest CGM reading |
| `GAP_FLAG_MIN` | 15 | Flag events with gaps exceeding this (minutes) |

## Project Structure

```
project/
├── source/                                    # Input data (do not modify)
│   ├── cgm_data/
│   │   └── CGM_<ParticipantID>.csv
│   ├── patient_extract0912_filtered_corrected.csv
│   └── MyFood24 ID Matched(Sheet1).csv
├── output/                                    # Generated results
│   ├── plots/
│   │   ├── <PID>_overview.png
│   │   └── global_summary.png
│   ├── corrected_meal_times_ALL.csv
│   ├── iauc_meal_events.csv
│   ├── patient_extract0912_realigned.csv
│   └── processing_report.csv
├── cgm_meal_realignment.py                    # Main pipeline
├── compute_iauc.py                            # iAUC computation from CGM + aligned meals
├── generate_realigned_source.py               # Produces realigned source CSV
├── CLOUDE_PROMPT.md                           # Prompt used to generate pipeline
├── USER_PROMPT.md                             # Original user requirements
└── README.md
```

## Input Files (in `source/`)

| File | Description |
|------|-------------|
| `patient_extract0912_filtered_corrected.csv` | Food diary export from MyFood24 with columns: Patient Id, Date, Time consumed at, Item added at, Meal, Food name, CHO, FAT, PROT, KCALS, TOTSUG, AOACFIB |
| `MyFood24 ID Matched(Sheet1).csv` | Mapping of Participant ID to MyFood24 ID |
| `cgm_data/CGM_<ParticipantID>.csv` | Per-participant CGM files with columns: isoDate, event_type, event_subtype, glucose (mmol/L), duration |

## Output Files (in `output/`)

| File | Description |
|------|-------------|
| `corrected_meal_times_ALL.csv` | All meal events with original and corrected times, time shift, confidence level, matched excursion details, and nutritional totals |
| `iauc_meal_events.csv` | Extension of `corrected_meal_times_ALL.csv` with 10 iAUC columns added after `meal_label`: baseline glucose, iAUC (mmol·min/L and mmol·h/L), peak glucose, time to peak, glucose at 120 min, CGM coverage metrics, and status |
| `patient_extract0912_realigned.csv` | Copy of the source diary with "Time consumed at" replaced by CGM-corrected times and a `time_shift_min` column added |
| `processing_report.csv` | Per-participant summary: match counts by confidence tier, excursion counts, mean/median shifts |
| `plots/<PID>_overview.png` | Per-participant CGM overlay plots showing reported vs corrected meal times with directional arrows |
| `plots/global_summary.png` | Aggregate statistics: CHO vs excursion rise, confidence breakdown, shift by meal type |

### Confidence Tiers

| Confidence | Meaning |
|------------|---------|
| `high` | Shift < 30 min |
| `medium` | Shift 30--90 min |
| `low_clamped` | Shift 90--180 min, capped to +/-30 min |
| `low_batch_override` | Batch day, shift accepted regardless of magnitude |
| `low_ampm_override` | 12h-format AM/PM resolved via CGM |
| `stacked` | Meal stacked onto an existing excursion (eaten during an active glucose response) |
| `no_match` | No suitable excursion found |
| `low_cho_no_match` | CHO < 5 g, no CGM match attempted |

## Per-Participant Plot Legend

Each subplot covers one day:

- **Steelblue line** -- raw CGM glucose trace
- **Dashed blue vertical line** -- reported meal time, with meal type label at the bottom (B=Breakfast, L=Lunch, D=Dinner, S=Snack, Dr=Drink)
- **Solid orange vertical line** -- corrected meal time
- **Orange arrow** -- direction and magnitude of the time shift (label shows shift in minutes)
- **Green triangle** -- glucose nadir (start of the excursion matched to the meal)

## Configuration

Key parameters at the top of `cgm_meal_realignment.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PHYSIOLOGICAL_LAG_MIN` | 20 | Expected delay (min) from eating to glucose rise onset |
| `MIN_EXCURSION_RISE_MMOL` | 0.8 | Minimum glucose rise to qualify as an excursion |
| `MAX_ALLOWABLE_SHIFT_MIN` | 180 | Maximum tolerated shift before rejection |
| `LOW_CONFIDENCE_CAP_MIN` | 30 | Clamp applied to low-confidence shifts |
| `CHO_THRESHOLD` | 5 | Minimum carbohydrate (g) to attempt CGM matching |
| `BATCH_ENTRY_GAP_MIN` | 30 | Timestamp gap defining separate entry sessions |
| `STACKING_WINDOW_MIN` | 45 | Window for stacking a meal onto an existing excursion |

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib

Install dependencies:

```bash
pip install pandas numpy scipy matplotlib
```

## Usage

1. Place the food diary CSV and ID mapping CSV in the `source/` directory.
2. Place all `CGM_*.csv` files in `source/cgm_data/`.
3. Update the `BASE` path in `cgm_meal_realignment.py` to point to the project root.
4. Run the main pipeline:

```bash
python cgm_meal_realignment.py
```

5. Compute iAUC for each meal event:

```bash
python compute_iauc.py
```

6. Generate the realigned source CSV:

```bash
python generate_realigned_source.py
```

All outputs are written to the `output/` directory.

## License

This project is provided for research purposes.
