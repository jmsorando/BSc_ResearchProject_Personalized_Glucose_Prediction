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

## Input Files

| File | Description |
|------|-------------|
| `patient_extract0912_filtered_corrected.csv` | Food diary export from MyFood24 with columns: Patient Id, Date, Time consumed at, Item added at, Meal, Food name, CHO, FAT, PROT, KCALS, TOTSUG, AOACFIB |
| `MyFood24 ID Matched(Sheet1).csv` | Mapping of Participant ID to MyFood24 ID |
| `cgm_data/CGM_<ParticipantID>.csv` | Per-participant CGM files with columns: isoDate, event_type, event_subtype, glucose (mmol/L), duration |

## Output Files

| File | Description |
|------|-------------|
| `corrected_meal_times_ALL.csv` | All meal events with original and corrected times, time shift, confidence level, matched excursion details, and nutritional totals |
| `processing_report.csv` | Per-participant summary: match counts by confidence tier, excursion counts, mean/median shifts |
| `plots/<PID>_overview.png` | Per-participant CGM overlay plots showing reported vs corrected meal times with directional arrows |
| `plots/global_summary.png` | Aggregate statistics: shift distribution, CHO vs excursion rise, confidence breakdown, shift by meal type |

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

1. Place the food diary CSV and ID mapping CSV in the same directory as the script.
2. Place all `CGM_*.csv` files in a `cgm_data/` subdirectory.
3. Update the `BASE` path in `cgm_meal_realignment.py` to point to the project directory.
4. Run:

```bash
python cgm_meal_realignment.py
```

Outputs are written to the same directory, with plots in the `plots/` subdirectory.

## License

This project is provided for research purposes.
