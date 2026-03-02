I'm working on an ML project predicting postprandial blood glucose responses (iAUC) using continuous glucose monitor data paired with food diary entries. I need you to build me a complete Python pipeline for correcting the food diary timestamps using the CGM data.

Here's the problem: participants wore a CGM sensor for about 2 weeks and also logged everything they ate in a food diary app called MyFood24. But the diary timestamps are unreliable — many people batch-entered their meals at the end of the day from memory, and some used 12-hour time format without AM/PM indicators so we can't tell if "7:10" means morning or evening. The CGM data shows clear glucose spikes after meals though, so we can use those to figure out when people actually ate.

The core idea is: detect glucose excursions in the CGM trace, then assign the food diary entries to those excursions. The CGM tells us WHEN eating happened, the diary tells us WHAT was eaten.

**My data files:**

1. `patient_extract0912_filtered_corrected.csv` — food diary with columns: Patient Id, Sex, Date (M/D/YYYY format), Time consumed at (H:MM or HH:MM), Item added at (ISO timestamp with timezone like +02:00), Meal (Breakfast/Lunch/Evening dinner/Snack/Drink), Food name, and macronutrients (CHO, FAT, PROT, KCALS, TOTSUG, AOACFIB plus ~150 others)

2. `CGM_<ParticipantID>.csv` files — one per participant, with isoDate (UTC), glucose (mmol/L, sometimes contains "Low" or "High" strings instead of numbers), readings every 5 minutes

3. `MyFood24 ID Matched(Sheet1).csv` — maps Participant ID (CGM ID like B389) to MyFood24 ID (diary Patient Id like 2615). Some IDs are compound like "F105/T189" or have trailing spaces

**The specific issues I need handled:**

**AM/PM confusion:** About 4 out of 70 participants used 12h format (max hour in their data is ≤12). Don't trust the meal label to resolve this — participants mislabel meals. Instead, generate both AM and PM candidates, then check which one has a matching CGM excursion nearby. Only fall back to the "Item added at" timestamp or meal label heuristics when there's no CGM evidence.

**Batch entry detection:** About 24% of participant-days are batch entries (entire day logged in 1-2 sessions). On these days the reported times are recalled from memory — the ordering is usually right (breakfast before lunch before dinner) but the absolute times can be way off. Use order-preserving assignment: match the k-th meal in reported order to the k-th excursion in chronological order. When there are more meals than excursions on batch days, prioritise meals by total sugars (TOTSUG) rather than total CHO, since sugar content better predicts which meals produced detectable acute excursions.

**Meal stacking:** A single glucose excursion often comes from multiple foods eaten within a ~45 min window (main course + dessert). Allow many-to-one matching — pick the best anchor meal per excursion, then let nearby unmatched meals stack onto it IF the CGM confirms glucose was still elevated at their reported time. Don't stack if glucose had returned to baseline between them.

**Noise-robust excursion detection:** On flat traces (glucose 5.1-5.5 mmol/L), sensor noise creates tiny dG/dt values that trigger false detections. Use two-stage detection: first flag candidates with the sensitive threshold, then confirm with a steep-rise check (dG/dt > 0.08). Re-anchor the nadir from the steep rise point, not the first noisy trigger — this prevents the nadir from locking onto a point 90 minutes too early.

**Cost function — use TOTSUG, not total CHO, for matching magnitude:** Total sugars (glucose, sucrose, maltose) are rapidly absorbed and produce the sharp, fast spikes that the excursion detection algorithm identifies. The cost function's CHO-rise correlation should use TOTSUG because the speed and magnitude of the postprandial peak are driven primarily by rapidly available carbohydrates. Total CHO includes starch (slow absorption, broad/flat curve) and fibre (blunts the response), so a high-CHO/low-sugar meal (e.g. lentils) barely registers as an excursion while a low-CHO/high-sugar meal (e.g. juice + biscuits) produces a sharp spike. Keep total CHO for the threshold filter (CHO_THRESHOLD) since even starchy meals eventually raise glucose.

**Conservative corrections:** Only apply high confidence (shift <30min) and medium confidence (30-90min) corrections fully. For low confidence matches (90-180min shift), CLAMP to ±30 min from the original reported time — UNLESS it's an AM/PM fix or a batch-entry day, which are the only legitimate reasons for large shifts. Anything >180min gets rejected entirely.

**Shift workers:** Some participants work nights, so don't assume any meal times based on clock — detect excursions 24/7, don't filter to "waking hours", and treat meal labels as descriptors of meal order not clock time. A "Breakfast" at 22:00 is valid.

**Cross-label merging:** Small mislabelled items (chocolate logged as "Lunch" at 20:29 when it's really post-dinner dessert) should get absorbed into nearby larger meal events before matching.

**Output should include:**
- `corrected_meal_times_ALL.csv` with one row per meal event: both original and corrected times, shift amount, confidence category, match type (anchor/stacked), excursion details, and metadata flags
- `processing_report.csv` with per-participant summary stats
- Per-participant visualisation plots showing CGM trace with reported vs corrected meal markers
- A global summary plot with shift distributions and CHO vs excursion magnitude

The script should be runnable as `python cgm_meal_realignment.py` with all CSV files in the same directory. Use pandas, numpy, scipy (savgol_filter), and matplotlib. Handle missing CGM files gracefully, and print progress per participant.