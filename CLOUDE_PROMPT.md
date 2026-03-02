# Original Prompt

The following is the original prompt used to generate the CGM-First Meal Time Re-Alignment Pipeline (`cgm_meal_realignment.py`).

---

# CGM-First Meal Time Re-Alignment Pipeline

## Philosophy

**The CGM is ground truth for WHEN eating happened. The food diary tells us WHAT was eaten. The pipeline assigns WHAT to WHEN.**

We do NOT try to "fix" diary timestamps then match to CGM. We detect eating events from the CGM first, then assign food diary entries to those events. This inverts the dependency: the CGM drives the timeline, the diary provides nutritional content.

---

## The Data Quality Landscape

Analysis of the full dataset (717 participant-days) reveals four scenarios:

| Scenario | Frequency | Description |
|---|---|---|
| **24h + real-time** | 71.1% | Participant logged throughout the day in 24h format. Times are approximate but roughly correct. |
| **24h + batch** | 23.4% | Participant logged entire day in 1-2 sessions but used 24h format. "Time consumed at" is recalled from memory -- ordering is usually correct, absolute times may be off. |
| **12h + real-time** | 4.9% | Participant logged throughout the day but used 12h format without AM/PM. Times are ambiguous (7:10 = 07:10 or 19:10?). |
| **12h + batch** | 0.6% | Worst case: batch-logged AND 12h ambiguity. |

50 out of ~70 participants have at least one batch-entry day. This is not an edge case -- it is a quarter of the data. The pipeline must handle all four scenarios robustly.

---

## Input Files (all in working directory)

### 1. `patient_extract0912_filtered_corrected.csv` -- Food diary
| Column | Description |
|---|---|
| `Patient Id` | MyFood24 numeric ID (e.g. `2615`) |
| `Sex` | Male/Female |
| `Date` | Date in `M/D/YYYY` format |
| `Time consumed at` | Reported time in `H:MM` or `HH:MM` -- may be 12h or 24h depending on participant |
| `Item added at` | ISO timestamp with timezone when entry was logged (system-generated, always accurate for logging time) |
| `Meal` | One of: `Breakfast`, `Lunch`, `Evening dinner`, `Snack`, `Drink` |
| `Food name` | Description of the food item |
| `CHO` | Carbohydrate grams |
| `FAT`, `PROT`, `KCALS`, `TOTSUG`, `AOACFIB` | Other macronutrients |
| ... | ~150 additional nutrient columns |

### 2. `CGM_<ParticipantID>.csv` -- One file per participant
| Column | Description |
|---|---|
| `isoDate` | ISO 8601 in **UTC** (e.g. `2025-06-19T14:35:51Z`) |
| `glucose` | mmol/L. May contain `Low` or `High` strings |

Readings at 5-minute intervals, ~10-day sensor duration.

### 3. `MyFood24 ID Matched(Sheet1).csv` -- ID mapping
| Column | Description |
|---|---|
| `Participant ID` | CGM ID (e.g. `B389`). May contain compound IDs (`F105/T189`) or trailing spaces. |
| `MyFood24 ID` | Food diary Patient Id (e.g. `2615`) |

---

## Configuration Constants

```python
# CGM processing
SMOOTHING_WINDOW = 5              # Savitzky-Golay window (odd)
SMOOTHING_POLY = 2                # Savitzky-Golay polynomial order
PHYSIOLOGICAL_LAG_MIN = 20        # minutes from eating to glucose rise start

# Excursion detection
MIN_EXCURSION_RISE_MMOL = 0.8     # minimum nadir-to-peak rise
MIN_SUSTAINED_RISE_MIN = 15       # sustained positive dG/dt duration
DGDT_THRESHOLD = 0.02             # mmol/L per 5-min interval
EXCURSION_MERGE_MIN = 25          # merge excursions closer than this
MIN_RISE_IN_WINDOW_MMOL = 0.3     # minimum absolute glucose rise within the sustained-rise window itself (noise filter)
STEEP_RISE_DGDT = 0.08            # mmol/L per 5-min -- threshold for "unambiguously real" rise (used for nadir re-anchoring)
NADIR_REANCHOR_LOOKBACK_MIN = 15  # when re-anchoring nadir, look back from steep rise start (not from first above-threshold reading)
# NOTE: No waking hours filter -- excursions are detected 24/7 to support shift workers

# Meal grouping
BATCH_ENTRY_GAP_MIN = 30          # max gap in "Item added at" to be same batch
CHO_THRESHOLD = 5                 # grams -- below this, meal won't produce detectable excursion

# Matching
MAX_ALLOWABLE_SHIFT_MIN = 180     # hard ceiling -- reject any match exceeding this
LOW_CONFIDENCE_CAP_MIN = 30       # for low-confidence matches: max allowed shift before clamping back toward reported time
STACKING_WINDOW_MIN = 45          # max gap between stacked meals
STACKING_ACTIVE_BUFFER_MIN = 30   # extra minutes past peak to extend active window

# AM/PM resolution
AMPM_CGM_SEARCH_WINDOW_MIN = 45   # search radius around each AM/PM candidate
```

---

## Pipeline Steps

**Execution order:** 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10

### Step 1: Load ID Mapping and Discover CGM Files

1. Read the matching sheet. For compound IDs (containing `/`), split and strip whitespace.
2. For each row, search for `CGM_<id>.csv` in the working directory.
3. Build mapping: `{MyFood24_ID: (participant_id, cgm_filepath)}`.
4. Report found vs missing.

### Step 2: Classify Each Participant's Data Quality

For each participant:

**2a. Detect time format (12h vs 24h):**
- Parse all "Time consumed at" hour values.
- If max hour across ALL entries <= 12 -> **12-hour format** (needs AM/PM resolution).
- If any hour > 12 -> **24-hour format** (no ambiguity).

**2b. Detect entry behaviour per day (real-time vs batch):**
- Sort "Item added at" timestamps for the day.
- Count distinct "add sessions" -- a new session starts when the gap between consecutive entries exceeds `BATCH_ENTRY_GAP_MIN`.
- Count distinct meal labels (excluding `Drink`).
- **Batch day** = <= 2 add sessions AND >= 3 meal labels.
- **Real-time day** = > 2 add sessions, OR < 3 meal labels.

Store per participant-day: `{time_format: "12h"/"24h", entry_type: "batch"/"realtime"}`.

### Step 3: Load and Preprocess CGM Data

For each participant's CGM file:

1. Parse `isoDate` as UTC.
2. Determine local timezone offset from the most common offset in that participant's `Item added at` timestamps (e.g. `+02:00`). Convert CGM to local.
3. Handle `"Low"` / `"High"` -> NaN, linearly interpolate gaps <= 6 readings (30 min). Mark gaps > 30 min as CGM dropout zones.
4. Apply Savitzky-Golay smoothing.
5. Compute `dG/dt` (5-min derivative) and 3-point rolling mean.

### Step 4: Detect Glucose Excursions (Eating Events)

For each day of CGM data (full 24 hours -- no waking-hours filter, as participants may be shift workers):

**The problem with naive detection:** On flat-but-noisy traces (glucose hovering at e.g. 5.1-5.5 mmol/L), sensor noise produces smoothed dG/dt values of 0.02-0.05 that pass the standard threshold. If three consecutive noisy readings happen to be positive, the algorithm triggers a false "sustained rise", anchors the nadir too early (e.g. 90 minutes before the real meal), and then finds a genuine peak much later -- producing an inflated excursion window with a wrong estimated meal time.

**Solution: Two-stage detection with steep-rise confirmation and nadir re-anchoring.**

**Stage 1 -- Candidate detection (sensitive, catches everything):**

1. Scan for regions where `dG/dt > DGDT_THRESHOLD` for >= `MIN_SUSTAINED_RISE_MIN` (3+ consecutive readings). These are **candidate** excursion starts.
2. For each candidate, measure the **actual glucose rise within the sustained-rise window itself**: `rise_in_window = glucose[end_of_window] - glucose[start_of_window]`.
3. **Noise filter:** If `rise_in_window < MIN_RISE_IN_WINDOW_MMOL` (0.3 mmol/L), discard the candidate. A real postprandial rise of 0.8+ mmol/L will always show >=0.3 mmol/L rise within any 15-minute sub-window of its ascending phase. Sensor noise on a flat trace will not.

**Stage 2 -- Steep-rise confirmation and nadir re-anchoring:**

For each candidate that passes Stage 1:

4. **Find the steep rise onset:** Scan forward from the candidate start to find the first reading where `dG/dt > STEEP_RISE_DGDT` (0.08 mmol/L per 5 min). This is where glucose is **unambiguously** rising due to a meal -- not noise. Call this `T_steep`.

5. **Re-anchor the nadir from the steep rise, not the first above-threshold reading:**
   - Search backward from `T_steep` for `NADIR_REANCHOR_LOOKBACK_MIN` (15 min) to find the local minimum.
   - This is the **true pre-meal nadir** -- the lowest point just before the real rise.
   - This prevents the scenario where a noisy dG/dt at 20:59 triggers detection but the real rise doesn't start until 22:04. The steep rise at 22:04 (dG/dt = 0.29) is found in Stage 2, and the nadir is re-anchored to ~21:49 instead of ~20:29.

6. **Find the peak:** Search forward from `T_steep` for up to 90 minutes to find the local maximum.

7. **Validate:** `rise = peak_glucose - nadir_glucose >= MIN_EXCURSION_RISE_MMOL`. Discard if not.

8. **Merge** excursions whose gap < `EXCURSION_MERGE_MIN`.

9. Compute `est_meal_time = nadir_time - PHYSIOLOGICAL_LAG_MIN`.

**Example -- the flat-trace false trigger case:**
```
Naive detection:                          Improved detection:
  dG/dt > 0.02 at 20:59 (noise)            dG/dt > 0.02 at 20:59 -> candidate
  -> nadir search from 20:59                rise_in_window (20:59-21:14) = 0.12 -> but let's say it passes
  -> nadir at 20:29 (5.15 mmol/L)           steep rise onset: scan forward -> T_steep at 22:04 (dG/dt = 0.29)
  -> peak at 22:54 (8.70)                   re-anchor nadir: search back 15 min from 22:04
  -> est_meal = 20:09                         -> nadir at 21:49 (5.31 mmol/L)
  -> shift from reported 21:45 = -96 min      -> peak at 22:54 (8.70)
  -> confidence: "low"                         -> est_meal = 21:29
                                              -> shift from reported 21:45 = -16 min
                                              -> confidence: "high"
```

**Each excursion defines an "eating window":**
```
active_start = nadir_time
active_end = peak_time + STACKING_ACTIVE_BUFFER_MIN
```
Any food consumed during this window contributed to this excursion.

**Output:** A list of eating events per day, each with: `excursion_id`, `est_meal_time`, `nadir_time`, `nadir_glucose`, `peak_time`, `peak_glucose`, `rise_mmol`, `active_start`, `active_end`.

### Step 5: Group Food Diary Into Meal Bundles

For each participant, for each day:

**5a. Initial grouping by label + batch proximity:**
1. Group entries by `Date` + `Meal` label.
2. Within each group, sub-group by `Item added at` proximity (gap > `BATCH_ENTRY_GAP_MIN` starts a new bundle).
3. Aggregate per bundle: `total_CHO`, `total_FAT`, `total_PROT`, `total_KCALS`, `total_TOTSUG`, `total_FIBRE`, `food_items`, `reported_time` (median of "Time consumed at"), `n_items`.

**5b. Cross-label temporal merging:**

After grouping, sort ALL bundles for the day by `reported_time`. Scan for small bundles (n_items <= 2 AND total_CHO < 25g) whose reported time falls within 60 minutes of a larger bundle. Absorb the small bundle into the larger one -- this catches mislabelled desserts/snacks that are really part of an adjacent meal.

### Step 6: Resolve 12-Hour AM/PM Ambiguity (CGM-First)

**Only applies to participants flagged as 12h format in Step 2a.**

For each meal bundle with an ambiguous time (hour 1-11):

1. Generate two candidates: `T_am = hour`, `T_pm = hour + 12`.
2. **If CGM data exists for this day and the bundle has CHO > CHO_THRESHOLD:**
   - For each candidate, search for a CGM excursion whose `est_meal_time` falls within +/- `AMPM_CGM_SEARCH_WINDOW_MIN`.
   - Score: `excursion_rise / (1 + time_distance_in_hours)`.
   - Pick the candidate with the higher score.
   - If tied or both zero -> go to fallback.
3. **Fallback (no CGM evidence):**
   - Use `Item added at` as anchor: prefer the candidate where `added_hour - candidate` is positive and smallest.
   - If AM gap > 8 hours -> prefer PM.
   - Final fallback: PM for Lunch/Snack/Drink (hour 1-9), AM for Breakfast (hour 5-11). Note: this fallback is only reached when there is no CGM evidence AND no `Item added at` signal -- it affects a negligible number of low-CHO entries (water, tea) where the AM/PM choice has no downstream impact on iAUC.
4. Record `ampm_resolved_by`: `"cgm"`, `"item_added"`, or `"fallback"`.

**The meal label is NEVER used as the primary AM/PM signal.** A "Breakfast" at `7:00` stays 07:00 only if the CGM agrees -- if there's a clear excursion at 19:30, it becomes 19:00 regardless of the label. This correctly handles shift workers whose "Breakfast" may be in the evening.

### Step 7: Detect Batch-Entry Days and Validate Reported Times

**This step specifically addresses Issue 3: meals logged in batch.**

For each day flagged as `entry_type = "batch"` in Step 2b:

**7a. Extract the day's meal sequence from the diary:**

Sort the day's bundles by `reported_time`. This gives the participant's **recalled order** of meals. On batch days, the absolute times may be wrong but the **relative ordering is usually correct** -- the participant knows they had breakfast before lunch before dinner, even if they can't remember exact times.

**7b. Extract the day's excursion sequence from the CGM:**

Sort the day's excursions by `est_meal_time`. These are the **true eating times**.

**7c. Order-preserving assignment:**

If the day has N significant meal bundles and M excursions:

1. **If N ~ M (+/-1):** The meal ordering and excursion ordering should correspond. Assign the k-th meal (by reported time order) to the k-th excursion (by time order). This respects the participant's recalled sequence while using the CGM for absolute timing.

2. **If N > M (more meals than excursions):** Some meals share an excursion (stacking). Assign the highest-TOTSUG meals first to excursions (since sugar content best predicts acute excursion magnitude), then stack remaining meals onto the nearest anchor (validated by the excursion's active window -- see Step 8c).

3. **If M > N (more excursions than meals):** Some excursions are non-meal (exercise, dawn phenomenon, stress). Match meals to excursions using the cost scoring, let extra excursions go unmatched.

**Note on meal labels:** Meal labels (Breakfast, Lunch, Evening dinner) are treated as participant-chosen descriptors, NOT as clock-time constraints. A shift worker's "Breakfast" at 22:00 or "Evening dinner" at 06:00 is valid -- it reflects the first/last meal of their day, not a fixed clock window. The pipeline never rejects or re-scores a match based on whether the corrected time "fits" the meal label.

### Step 8: Assign Meals to Excursions (Anchor + Stacking)

This is the core matching step, applied to ALL days (batch and real-time).

**8a. Separate by CHO significance:**
- **Significant** (total_CHO > `CHO_THRESHOLD`): need excursion assignment.
- **Insignificant** (total_CHO <= threshold): keep reported time, `confidence = "low_cho_no_match"`.

**8b. Pass 1 -- Anchor matching (best meal per excursion):**

For each excursion (sorted by time), find the best anchor meal:

1. **For real-time days:** Score each unmatched significant meal:
   ```
   cost = W_time * |reported_time - est_meal_time| / 3600
        + W_sug  * |norm_TOTSUG - norm_rise|
   ```
   Where `W_time = 2.0`, `W_sug = 1.0`. Total sugars (TOTSUG) is used instead of total CHO because simple sugars drive the acute, sharp postprandial spikes that the excursion detector identifies. Total CHO includes starch and fibre, which produce slower, flatter responses that correlate poorly with excursion rise magnitude. CHO is still used for the threshold filter (`CHO_THRESHOLD`) since even starchy meals eventually raise glucose.

2. **For batch days:** Use the **order-preserving assignment** from Step 7c instead of cost-based matching. The k-th meal in the recalled sequence maps to the k-th excursion.

3. **Hard constraint:** Any match where `|shift| > MAX_ALLOWABLE_SHIFT_MIN` -> infinite cost. Never allow.

4. Assign the lowest-cost meal as the **anchor** for that excursion.
   - `corrected_time = est_meal_time`
   - `match_type = "anchor"`

**8c. Pass 2 -- Physiological stacking (remaining meals onto anchors):**

For every significant meal NOT assigned as an anchor:

1. Find the nearest anchor meal by reported time. Compute `gap = |unmatched_reported_time - anchor_corrected_time|`.
2. **If gap <= STACKING_WINDOW_MIN:**
   - Validate against CGM: does the meal's reported time fall within the excursion's active window (`active_start` to `active_end`)?
   - **If YES -> Stack:**
     - `corrected_time` = meal's own reported time (it's already plausible within the excursion window)
     - `confidence = "stacked"`
     - `match_type = "stacked"`
     - `stacked_onto_event_id` = anchor's event_id
     - `excursion_id` = shared with anchor
   - **If NO -> Do not stack.** The CGM shows glucose had returned to baseline -- this is a separate eating event that just lacks its own detectable excursion.
3. **If gap > STACKING_WINDOW_MIN:** No stacking. Meal gets `confidence = "no_match"`, keeps reported time.

**Why CGM validation matters for stacking:** Without it, a snack eaten 40 minutes after dinner would always stack -- even if the CGM shows glucose returned to baseline at minute 25 and there's a fresh small rise at minute 35 (meaning the snack DID cause its own separate excursion that was below the detection threshold). The CGM check ensures we only stack when it's physiologically justified.

**8d. Pass 3 -- Conservative correction policy:**

The pipeline applies a **conservative-by-default** approach: only high and medium confidence matches are applied as-is. Low confidence matches are clamped to prevent drastic, hard-to-justify time shifts. Two specific scenarios override this conservatism because the reported times are known to be fundamentally unreliable.

For each anchor-matched meal, compute the raw shift:
```
raw_shift_min = corrected_time - reported_time
```

Then apply the correction policy:

```
1. HARD REJECTION (always applies):
   IF |raw_shift_min| > MAX_ALLOWABLE_SHIFT_MIN (180 min):
       -> Revert to reported time
       -> confidence = "rejected_implausible_shift"
       -> STOP, no further processing

2. HIGH CONFIDENCE (|raw_shift| < 30 min):
   -> Apply correction as-is
   -> corrected_time = est_meal_time
   -> confidence = "high"

3. MEDIUM CONFIDENCE (30 <= |raw_shift| < 90 min):
   -> Apply correction as-is
   -> corrected_time = est_meal_time
   -> confidence = "medium"

4. LOW CONFIDENCE (90 <= |raw_shift| < 180 min):
   Check for exceptions:

   EXCEPTION A -- AM/PM confusion (12h format participant):
       IF time_format == "12h" AND ampm_resolved_by == "cgm":
       -> Apply correction as-is (the large shift IS the AM/PM fix)
       -> corrected_time = est_meal_time
       -> confidence = "low_ampm_override"

   EXCEPTION B -- Batch entry day:
       IF batch_day == True:
       -> Apply correction as-is (reported times are recalled from memory, unreliable)
       -> corrected_time = est_meal_time
       -> confidence = "low_batch_override"

   DEFAULT -- No exception applies:
       -> CLAMP the correction to +/-LOW_CONFIDENCE_CAP_MIN (30 min) from reported time
       -> corrected_time = reported_time + clamp(raw_shift, -30, +30)
       -> confidence = "low_clamped"
       -> Log: "Low-confidence match clamped: original shift <N>min -> capped to +/-30min"
```

**Why this works:**
- High/medium matches are well-supported by the CGM -- the excursion is close to the reported time and the correction is modest. Apply freely.
- Low-confidence matches on **normal real-time days** often indicate the algorithm matched to the wrong excursion or a marginal fluctuation. Clamping to +/-30 min prevents catastrophic corrections while still allowing minor refinement.
- Low-confidence matches on **AM/PM days** are expected to have large shifts (~12 hours) -- that's the whole point of AM/PM resolution. The CGM confirmed the correct interpretation, so the large shift is justified.
- Low-confidence matches on **batch days** are expected to have large shifts because the participant entered recalled times from memory, often hours off. The order-preserving assignment (Step 7c) provides structural validation even when the absolute shift is large.

### Step 9: Assign Confidence Scores

**Summary of all confidence categories:**

```
Anchor matches (applied corrections):
  "high"                        -- |shift| < 30 min, correction applied as-is
  "medium"                      -- |shift| 30-90 min, correction applied as-is
  "low_clamped"                 -- |shift| 90-180 min on a normal day, clamped to +/-30 min
  "low_ampm_override"           -- |shift| 90-180 min BUT 12h participant with CGM-confirmed AM/PM fix
  "low_batch_override"          -- |shift| 90-180 min BUT batch-entry day with order-preserving assignment
  "rejected_implausible_shift"  -- |shift| > 180 min, reverted to reported time

Stacked meals:
  "stacked"                     -- reported time kept, CGM-validated within excursion active window

Unmatched meals:
  "no_match"                    -- significant CHO, no plausible excursion or stacking target
  "low_cho_no_match"            -- CHO <= threshold (water, tea, black coffee)
  "no_cgm_data"                 -- no CGM data for that day
  "flat_trace"                  -- day had <2 excursions for 3+ significant meals
  "cgm_gap"                     -- meal falls in a CGM dropout period (>30 min gap)
```

**Confidence hierarchy for downstream filtering:**
```
Most trustworthy -> least trustworthy:
  high > medium > stacked > low_ampm_override > low_batch_override > low_clamped > no_match > rejected
```

For ML training, the recommended minimum inclusion threshold is `medium` or above (plus `stacked` meals which inherit their anchor's excursion). The `low_*` categories can be included with appropriate caution or used for validation.

**Additional flags (not confidence, but metadata):**
```
batch_day = True/False        -- this day was a batch-entry day
ampm_resolved_by              -- "cgm" / "item_added" / "fallback" / "unambiguous" / "24h_format"
```

### Step 10: Produce Outputs

#### 10a. `corrected_meal_times_ALL.csv` -- Main output

One row per meal event:

| Column | Description |
|---|---|
| `participant_id` | CGM participant ID |
| `myfood24_id` | Food diary patient ID |
| `date` | Date of meal |
| `meal_label` | Original diary label (Breakfast/Lunch/Evening dinner/Snack/Drink) |
| `food_items` | Semicolon-separated food names |
| `n_items` | Count of diary entries in this bundle |
| `total_CHO` | Carbohydrates (g) |
| `total_FAT` | Fat (g) |
| `total_PROT` | Protein (g) |
| `total_KCALS` | Kilocalories |
| `total_TOTSUG` | Total sugars (g) |
| `total_FIBRE` | Fibre (g) |
| `reported_time` | Parsed time from diary (after AM/PM resolution if applicable) |
| `corrected_time` | CGM-aligned time |
| `time_shift_min` | Shift in minutes (positive = later than reported) |
| `confidence` | See Step 9 |
| `match_type` | "anchor" / "stacked" / null |
| `excursion_id` | ID of matched excursion (shared between anchor + stacked) |
| `stacked_onto_event_id` | If stacked: anchor's event_id |
| `excursion_rise_mmol` | Rise magnitude of matched excursion |
| `excursion_peak_mmol` | Peak glucose of matched excursion |
| `nadir_time` | Excursion nadir timestamp |
| `peak_time` | Excursion peak timestamp |
| `batch_day` | True if this day was batch-entry |
| `time_format` | "12h" / "24h" |
| `ampm_resolved_by` | How AM/PM was resolved |
| `tz_offset` | Timezone offset applied |
| `event_id` | Unique event identifier |

#### 10b. `processing_report.csv` -- Per-participant summary

| Column | Description |
|---|---|
| `participant_id` | CGM ID |
| `myfood24_id` | Diary ID |
| `cgm_file_found` | True/False |
| `time_format` | 12h / 24h |
| `tz_offset` | Timezone offset |
| `diary_days` / `cgm_days` / `overlap_days` | Day counts |
| `batch_days` / `realtime_days` | Entry type breakdown |
| `total_meal_events` | Total bundles |
| `matched_high` / `matched_medium` | Anchor match confidence counts |
| `low_clamped` / `low_ampm_override` / `low_batch_override` | Low-confidence breakdown |
| `stacked` | Stacked meal count |
| `no_match` / `rejected_shifts` | Unmatched counts |
| `low_cho` / `no_cgm` | Low-CHO and no-data counts |
| `excursions_detected` | Total CGM excursions |
| `mean_shift_min` / `median_shift_min` | Correction statistics |

#### 10c. `plots/<participant_id>_overview.png` -- Per-participant visualisation

Multi-day panel:
- Smoothed CGM trace with excursion shading
- Blue dashed lines = reported meal times
- Orange solid lines = corrected meal times
- Purple arrows = shift direction
- Labels: meal name, CHO, shift, confidence
- Stacked meals shown grouped (bracket notation)
- Batch-entry days marked with a banner

#### 10d. `plots/global_summary.png` -- Dataset-wide summary

4 panels:
1. Histogram of time shifts (all participants pooled)
2. CHO vs excursion rise scatter (coloured by confidence)
3. Confidence category distribution (pie/bar)
4. Mean shift by meal type

---

## Technical Requirements

- Python 3.10+
- Libraries: `pandas`, `numpy`, `scipy` (for `savgol_filter`), `matplotlib`
- Handle gracefully: missing CGM files, compound participant IDs (`F105/T189`), mixed timezone offsets (`+01:00` / `+02:00`), `Low`/`High` glucose strings, empty diary days
- Progress logging to stdout per participant
- Runnable as: `python cgm_meal_realignment.py` with all CSVs in the same directory
- Output directories created automatically (`plots/`)

---

## Edge Cases

1. **Overnight eating is legitimate:** Participants may be shift workers. A meal at 02:00 AM is a real meal, not a data error. The pipeline detects excursions 24/7 and makes no assumptions about "normal" meal times. Meal labels like "Breakfast" and "Evening dinner" are treated as participant-chosen descriptors of meal order, not clock-time indicators.

2. **No CGM-diary overlap:** Some participants wore CGM on different weeks than diary. All meals get `no_cgm_data`.

3. **Multiple MyFood24 IDs per participant:** e.g. B101 -> 2339, 2340. Process both IDs under the same participant.

4. **Cross-midnight meals:** Entries between 00:00-05:00 may be logged under either the current or previous calendar date. When matching, search both the current and previous day's CGM excursions for the best match. The cost function handles this naturally -- the correct excursion will be closest in time regardless of which calendar date it falls on.

5. **Very flat CGM:** Non-diabetic participants with minimal variability. If <2 excursions detected on a day with >=3 significant meals -> `flat_trace` for all meals, keep reported times.

6. **Multi-course meals (stacking):** Dinner main course at 19:00 + dessert at 20:15 -> single excursion. Pass 2 stacking handles this: dessert stacks onto dinner anchor if CGM confirms glucose still elevated.

7. **CGM dropout:** Gaps >30 min -> don't interpolate. Meals in gap periods -> `cgm_gap`.

8. **Mislabelled meals:** Chocolate logged as "Lunch" at 20:29 (actually evening dessert). Cross-label merging (Step 5b) may absorb it. If not, stacking (Step 8c) links it to the nearby dinner excursion. If neither catches it, the MAX_ALLOWABLE_SHIFT_MIN constraint prevents a nonsensical distant match.

9. **Batch entry with wrong recall:** On batch days, participant may recall "Breakfast at 8:30" when they actually ate at 9:15. The order-preserving assignment (Step 7d -> 8b) uses CGM excursion timing rather than the recalled absolute time, correcting this naturally.

10. **Ambiguous 12h time with no CGM evidence:** e.g. water at `3:00` -- no excursion to resolve AM/PM. Falls back to `Item added at` anchor, then meal-label heuristic. Low-CHO items like water are flagged `low_cho_no_match` anyway, so the AM/PM choice doesn't affect downstream iAUC.

---

## Downstream ML Note

When computing iAUC features for a given excursion, **aggregate the nutritional profiles of ALL meals sharing that excursion_id** (anchor + all stacked meals). This gives the model the complete input that produced the glucose response.

Example:
```
excursion_id: EXC_042
  anchor:  Evening dinner (45g CHO, 26g FAT, 18g PROT, 835 kcal)
  stacked: Kefir + Blueberries (35g CHO, 4g FAT, 12g PROT, 250 kcal)
  stacked: Chocolate (18g CHO, 8g FAT, 2g PROT, 165 kcal)

-> Model input for EXC_042: 98g CHO, 38g FAT, 32g PROT, 1250 kcal
-> Model target: iAUC from nadir to baseline return
```
