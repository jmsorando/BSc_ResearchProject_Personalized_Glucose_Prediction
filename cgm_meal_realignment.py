#!/usr/bin/env python3
"""CGM-First Meal Time Re-Alignment Pipeline v3"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd, numpy as np, warnings, os
from scipy.signal import savgol_filter
from pathlib import Path
from datetime import datetime, timedelta, date as date_type
from collections import Counter, defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
BASE = Path(r"C:\Users\Jose Miguel Sorando\Documents\RP Cleaning 4 (Claude)")
SRC  = BASE / "source"
OUT  = BASE / "output"

# ── Config ──────────────────────────────────────────────────────────
SMOOTHING_WINDOW        = 5
SMOOTHING_POLY          = 2
PHYSIOLOGICAL_LAG_MIN   = 20
MIN_EXCURSION_RISE_MMOL = 0.8
MIN_SUSTAINED_READINGS  = 3       # 3 readings = 15 min
DGDT_THRESHOLD          = 0.02
EXCURSION_MERGE_MIN     = 25
MIN_RISE_IN_WINDOW_MMOL = 0.3
STEEP_RISE_DGDT         = 0.08
NADIR_REANCHOR_LOOKBACK = 3       # readings (15 min)
BATCH_ENTRY_GAP_MIN     = 30
CHO_THRESHOLD           = 5
CROSS_LABEL_MERGE_MIN   = 60
MAX_ALLOWABLE_SHIFT_MIN = 180
LOW_CONFIDENCE_CAP_MIN  = 30
STACKING_WINDOW_MIN     = 45
STACKING_ACTIVE_BUF_MIN = 30
W_TIME = 2.0;  W_SUG = 1.0
AMPM_SEARCH_WIN_MIN     = 45

# ── Helpers ─────────────────────────────────────────────────────────
def _discover_files():
    """Find actual filenames on disk."""
    diary_f = mapping_f = None
    for f in SRC.iterdir():
        if not f.is_file() or f.suffix != ".csv": continue
        n = f.name.lower()
        if "patient_extract" in n and "corrected" in n: diary_f = f.name
        if "myfood24" in n and "matched" in n:          mapping_f = f.name
    # fallbacks
    if diary_f is None: diary_f = "patient_extract0912_filtered_corrected.csv"
    if mapping_f is None: mapping_f = "MyFood24 ID Matched(Sheet1).csv"
    return diary_f, mapping_f

def _get_tz_offset(diary_sub):
    raw = diary_sub["Item added at"].dropna().astype(str)
    offsets = []
    for s in raw:
        try:
            idx = max(s.rfind("+"), s.rfind("-"))
            if idx < 10: continue
            off = s[idx:]
            sign = 1 if off[0] == "+" else -1
            pts = off[1:].split(":")
            td = timedelta(hours=sign*int(pts[0]),
                           minutes=sign*int(pts[1]) if len(pts)>1 else 0)
            offsets.append(td)
        except: pass
    if offsets:
        return Counter(offsets).most_common(1)[0][0]
    return timedelta(0)

def _td_to_str(td):
    s = int(td.total_seconds())
    sign = "+" if s >= 0 else "-"
    s = abs(s)
    return f"{sign}{s//3600:02d}:{(s%3600)//60:02d}"

def _parse_meal_time(row, d):
    try:
        pts = str(row["Time consumed at"]).split(":")
        h, m = int(pts[0]), int(pts[1]) if len(pts)>1 else 0
        return datetime(d.year, d.month, d.day, h, m)
    except: return None

# ═══════════════════════════════════════════════════════════════════
#  STEP 1
# ═══════════════════════════════════════════════════════════════════
def step1():
    diary_f, mapping_f = _discover_files()
    print("="*72); print("STEP 1: Load ID Mapping and Discover CGM Files"); print("="*72)
    print(f"  Mapping: {mapping_f}")

    mp = pd.read_csv(SRC/mapping_f, dtype=str)
    mp.columns = mp.columns.str.strip()

    mapping = {}
    for _, row in mp.iterrows():
        raw = str(row["Participant ID"]).strip()
        mf24 = str(row["MyFood24 ID"]).strip()
        pids = [p.strip() for p in raw.replace("/",",").split(",") if p.strip()]
        cgm_path, chosen = None, pids[0] if pids else raw
        for pid in pids:
            c = SRC/"cgm_data"/f"CGM_{pid}.csv"
            if c.exists(): cgm_path, chosen = c, pid; break
        if mf24 not in mapping or (mapping[mf24][1] is None and cgm_path):
            mapping[mf24] = (chosen, cgm_path)

    found = sum(1 for _,(_,p) in mapping.items() if p)
    print(f"  {found} with CGM, {len(mapping)-found} without")

    print(f"\n  Loading diary: {diary_f}")
    diary = pd.read_csv(SRC/diary_f, low_memory=False)
    diary["Patient Id"] = diary["Patient Id"].astype(str).str.strip()
    print(f"  {len(diary)} diary entries")
    return mapping, diary

# ═══════════════════════════════════════════════════════════════════
#  STEP 2
# ═══════════════════════════════════════════════════════════════════
def step2(diary, mapping):
    print("\n"+"="*72); print("STEP 2: Classify Data Quality"); print("="*72)
    time_fmts = {}
    day_info  = {}
    p12 = []

    mf24s = sorted(set(diary["Patient Id"].unique()) & set(mapping.keys()))
    for mf24 in mf24s:
        sub = diary[diary["Patient Id"]==mf24]
        hours = sub["Time consumed at"].str.split(":").str[0]
        hours = pd.to_numeric(hours, errors="coerce").dropna()
        tf = "12h" if len(hours)>0 and hours.max()<=12 else "24h"
        time_fmts[mf24] = tf
        if tf=="12h": p12.append(mf24)

        dates_col = pd.to_datetime(sub["Date"], format="mixed", dayfirst=False).dt.date
        for d in dates_col.unique():
            ds = sub[dates_col==d]
            added = pd.to_datetime(ds["Item added at"], errors="coerce", utc=True).dropna().sort_values()
            n_sess = 1
            if len(added)>1:
                n_sess = 1 + (added.diff().dt.total_seconds()/60 > BATCH_ENTRY_GAP_MIN).sum()
            n_labels = ds[ds["Meal"]!="Drink"]["Meal"].nunique()
            day_info[(mf24,d)] = "batch" if n_sess<=2 and n_labels>=3 else "realtime"

    nb = sum(1 for v in day_info.values() if v=="batch")
    nr = sum(1 for v in day_info.values() if v=="realtime")
    print(f"  12h-format participants ({len(p12)}): {p12}")
    print(f"  {nr} real-time days, {nb} batch days")
    return time_fmts, day_info

# ═══════════════════════════════════════════════════════════════════
#  STEP 3
# ═══════════════════════════════════════════════════════════════════
def step3(cgm_path, diary_sub):
    cgm = pd.read_csv(cgm_path, low_memory=False)
    if "event_type" in cgm.columns:
        cgm = cgm[cgm["event_type"]=="EGV"].copy()
    cgm["ts_utc"] = pd.to_datetime(cgm["isoDate"], utc=True, errors="coerce")
    cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")
    cgm = cgm.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)

    tz = _get_tz_offset(diary_sub)
    tz_str = _td_to_str(tz)
    cgm["ts_local"] = (cgm["ts_utc"]+tz).dt.tz_localize(None)
    cgm["date_local"] = cgm["ts_local"].dt.date

    # interpolate ≤6 missing
    cgm["glucose"] = cgm["glucose"].interpolate(method="linear", limit=6)

    # identify gaps >30 min
    dt = cgm["ts_local"].diff().dt.total_seconds()/60
    gap_mask = dt>30
    gaps = []
    for i in np.where(gap_mask)[0]:
        gaps.append((cgm["ts_local"].iloc[i-1], cgm["ts_local"].iloc[i]))

    # segment boundaries for smoothing
    gi = sorted(np.where(gap_mask)[0])
    bounds = [0]+gi+[len(cgm)]

    cgm["gluc_s"] = np.nan
    for j in range(len(bounds)-1):
        s,e = bounds[j], bounds[j+1]
        idx = cgm.index[s:e]
        v = cgm.loc[idx,"glucose"].values.copy()
        ok = ~np.isnan(v)
        nv = ok.sum()
        if nv >= SMOOTHING_WINDOW:
            w = SMOOTHING_WINDOW
            sv = savgol_filter(v[ok], w, SMOOTHING_POLY)
            out = np.full(len(v), np.nan); out[ok]=sv
            cgm.loc[idx,"gluc_s"] = out
        elif nv>0:
            cgm.loc[idx,"gluc_s"] = v

    # dG/dt per segment
    cgm["dgdt"] = np.nan
    for j in range(len(bounds)-1):
        s,e = bounds[j], bounds[j+1]
        idx = cgm.index[s:e]
        if len(idx)<2: continue
        gs = cgm.loc[idx,"gluc_s"]
        dt_seg = cgm.loc[idx,"ts_local"].diff().dt.total_seconds()/300
        cgm.loc[idx,"dgdt"] = gs.diff()/dt_seg

    cgm["dgdt_rm"] = cgm["dgdt"].rolling(3, center=True, min_periods=1).mean()
    return cgm, gaps, tz_str, tz

# ═══════════════════════════════════════════════════════════════════
#  STEP 4
# ═══════════════════════════════════════════════════════════════════
def step4(cgm, mf24):
    exc_by_date = {}
    ctr = [0]
    for d, dc in cgm.groupby("date_local"):
        dc = dc.sort_values("ts_local").reset_index(drop=True)
        exc_by_date[d] = _detect_day(dc, mf24, ctr)
    return exc_by_date

def _detect_day(cg, mf24, ctr):
    n = len(cg)
    if n<5: return []
    dg = cg["dgdt_rm"].values
    gl = cg["gluc_s"].values
    ts = cg["ts_local"].values
    candidates = []

    # Stage 1: sustained rises
    i=0
    while i<n:
        if np.isnan(dg[i]) or dg[i]<=DGDT_THRESHOLD: i+=1; continue
        rs=i
        while i<n and not np.isnan(dg[i]) and dg[i]>DGDT_THRESHOLD: i+=1
        re=i-1
        if re-rs+1 < MIN_SUSTAINED_READINGS: continue
        g0,g1 = gl[rs], gl[re]
        if np.isnan(g0) or np.isnan(g1): continue
        if g1-g0 < MIN_RISE_IN_WINDOW_MMOL: continue
        candidates.append((rs,re))

    # Stage 2
    excs = []
    for rs,re in candidates:
        # steep rise
        ts_idx = None
        for k in range(rs, min(n, rs+18)):
            if not np.isnan(dg[k]) and dg[k]>STEEP_RISE_DGDT:
                ts_idx=k; break

        if ts_idx is not None:
            lb = max(0, ts_idx-NADIR_REANCHOR_LOOKBACK)
            seg = gl[lb:ts_idx+1]
            if np.all(np.isnan(seg)): continue
            ni = lb + int(np.nanargmin(seg))
            ps = ts_idx
        else:
            lb = max(0, rs-3)
            seg = gl[lb:rs+1]
            if np.all(np.isnan(seg)): continue
            ni = lb + int(np.nanargmin(seg))
            ps = rs

        # peak
        pe = min(n, ps+18)
        seg_p = gl[ps:pe]
        if np.all(np.isnan(seg_p)): continue
        pi = ps + int(np.nanargmax(seg_p))

        if np.isnan(gl[ni]) or np.isnan(gl[pi]): continue
        rise = gl[pi]-gl[ni]
        if rise < MIN_EXCURSION_RISE_MMOL: continue

        ctr[0]+=1
        excs.append({
            "excursion_id": f"EXC_{ctr[0]:04d}",
            "nadir_time": pd.Timestamp(ts[ni]),
            "peak_time":  pd.Timestamp(ts[pi]),
            "nadir_glucose": float(gl[ni]),
            "peak_glucose":  float(gl[pi]),
            "rise_mmol": float(rise),
            "est_meal_time": pd.Timestamp(ts[ni]) - pd.Timedelta(minutes=PHYSIOLOGICAL_LAG_MIN),
            "active_start":  pd.Timestamp(ts[ni]),
            "active_end":    pd.Timestamp(ts[pi]) + pd.Timedelta(minutes=STACKING_ACTIVE_BUF_MIN),
        })

    # merge close excursions
    if len(excs)>1:
        merged=[excs[0]]
        for e in excs[1:]:
            p=merged[-1]
            gap=(e["nadir_time"]-p["peak_time"]).total_seconds()/60
            if gap<EXCURSION_MERGE_MIN:
                if e["peak_glucose"]>p["peak_glucose"]:
                    p["peak_time"]=e["peak_time"]; p["peak_glucose"]=e["peak_glucose"]
                p["rise_mmol"]=p["peak_glucose"]-p["nadir_glucose"]
                p["active_end"]=p["peak_time"]+pd.Timedelta(minutes=STACKING_ACTIVE_BUF_MIN)
            else: merged.append(e)
        excs=merged
    return excs

# ═══════════════════════════════════════════════════════════════════
#  STEP 5
# ═══════════════════════════════════════════════════════════════════
def step5(diary_sub, mf24, tz_offset):
    diary_sub = diary_sub.copy()
    diary_sub["_date"] = pd.to_datetime(diary_sub["Date"], format="mixed", dayfirst=False).dt.date
    diary_sub["_added"] = pd.to_datetime(diary_sub["Item added at"], errors="coerce", utc=True)

    bundles_by_date = {}
    for d, ds in diary_sub.groupby("_date"):
        bundles = []
        for label, ls in ds.groupby("Meal"):
            ls = ls.sort_values("_added")
            cur = [ls.iloc[0]]
            for j in range(1, len(ls)):
                r = ls.iloc[j]
                prev_a = cur[-1]["_added"]
                cur_a  = r["_added"]
                gap = abs((cur_a-prev_a).total_seconds()/60) if pd.notna(cur_a) and pd.notna(prev_a) else 0
                if gap > BATCH_ENTRY_GAP_MIN:
                    bundles.append(_agg(cur, label, d))
                    cur = [r]
                else:
                    cur.append(r)
            if cur: bundles.append(_agg(cur, label, d))

        # cross-label merge
        bundles.sort(key=lambda b: b["reported_time"] if b["reported_time"] else datetime.min)
        absorbed = set()
        final = []
        for i,b in enumerate(bundles):
            if i in absorbed: continue
            for j,o in enumerate(bundles):
                if j<=i or j in absorbed: continue
                if o["n_items"]<=2 and o["total_CHO"]<25 and b["n_items"]>o["n_items"]:
                    if b["reported_time"] and o["reported_time"]:
                        gap=abs((o["reported_time"]-b["reported_time"]).total_seconds()/60)
                        if gap<=CROSS_LABEL_MERGE_MIN:
                            b["n_items"]+=o["n_items"]
                            for col in NUTRIENT_COLS:
                                k = f"total_{col}"
                                b[k] = b.get(k, 0) + o.get(k, 0)
                            b["food_items"]+="; "+o["food_items"]
                            absorbed.add(j)
            final.append(b)
        bundles_by_date[d] = final
    return bundles_by_date

def _agg(rows, label, d):
    if isinstance(rows, list):
        df = pd.DataFrame(rows) if not isinstance(rows[0], dict) else pd.DataFrame(rows)
    else:
        df = rows
    times = []
    for _, r in df.iterrows():
        t = _parse_meal_time(r, d)
        if t: times.append(t)
    rep = sorted(times)[len(times)//2] if times else None

    added_list = df["Item added at"].tolist() if "Item added at" in df.columns else []
    result = {
        "meal_label": label, "date": d, "reported_time": rep,
        "n_items": len(df),
        "food_items": "; ".join(df["Food name"].astype(str).tolist()),
        "_added_list": added_list,
    }
    for col in NUTRIENT_COLS:
        if col in df.columns:
            result[f"total_{col}"] = pd.to_numeric(df[col], errors="coerce").sum()
        else:
            result[f"total_{col}"] = 0.0
    return result

# ═══════════════════════════════════════════════════════════════════
#  STEP 6
# ═══════════════════════════════════════════════════════════════════
def step6(bundles_by_date, exc_by_date, tf):
    for d, bundles in bundles_by_date.items():
        day_exc = exc_by_date.get(d, [])
        for b in bundles:
            if tf != "12h":
                b["ampm_resolved_by"] = "24h_format"; continue
            rt = b["reported_time"]
            if not rt: b["ampm_resolved_by"]="fallback"; continue
            h = rt.hour
            if h==0 or h==12: b["ampm_resolved_by"]="unambiguous"; continue

            t_am, t_pm = rt, rt.replace(hour=h+12)

            # CGM scoring
            if day_exc and b["total_CHO"]>CHO_THRESHOLD:
                sa = _ampm_sc(t_am, day_exc); sp = _ampm_sc(t_pm, day_exc)
                if sa>0 or sp>0:
                    if sp>sa: b["reported_time"]=t_pm
                    b["ampm_resolved_by"]="cgm"; continue

            # item_added fallback
            added = b.get("_added_list",[])
            ap = pd.to_datetime(pd.Series(added), errors="coerce", utc=True).dropna()
            if len(ap)>0:
                ah = ap.iloc[0].hour
                gam, gpm = ah-h, ah-(h+12)
                if gam>=0 and (gpm<0 or gam<abs(gpm)):
                    b["ampm_resolved_by"]="item_added"; continue
                if gpm>=0:
                    b["reported_time"]=t_pm; b["ampm_resolved_by"]="item_added"; continue
                if gam>8:
                    b["reported_time"]=t_pm; b["ampm_resolved_by"]="item_added"; continue

            # label fallback
            lab = b["meal_label"]
            if lab=="Breakfast" and 5<=h<=11: b["ampm_resolved_by"]="fallback"
            elif lab in ("Lunch","Snack","Drink") and 1<=h<=9:
                b["reported_time"]=t_pm; b["ampm_resolved_by"]="fallback"
            elif lab=="Evening dinner":
                b["reported_time"]=t_pm; b["ampm_resolved_by"]="fallback"
            else: b["ampm_resolved_by"]="fallback"
    return bundles_by_date

def _ampm_sc(cand, excs):
    best=0
    for e in excs:
        d=abs((cand-e["est_meal_time"]).total_seconds()/60)
        if d<=AMPM_SEARCH_WIN_MIN:
            best=max(best, e["rise_mmol"]/(1+d/60))
    return best

# ═══════════════════════════════════════════════════════════════════
#  STEP 8
# ═══════════════════════════════════════════════════════════════════
def step8(bundles_by_date, exc_by_date, day_info, tf, mf24, gaps, cgm_dates):
    results = []; ectr=[0]

    for d, bundles in sorted(bundles_by_date.items()):
        et = day_info.get((mf24,d),"realtime")
        is_batch = et=="batch"
        day_exc = exc_by_date.get(d,[])

        # cross-midnight for real-time only
        if not is_batch:
            prev = exc_by_date.get(d-timedelta(days=1),[])
            ext_exc = [e for e in prev if e["est_meal_time"].hour>=20] + day_exc
        else:
            ext_exc = list(day_exc)

        has_cgm = d in cgm_dates
        sig = [(i,b) for i,b in enumerate(bundles) if b["total_CHO"]>CHO_THRESHOLD]
        is_flat = has_cgm and len(day_exc)<2 and len(sig)>=3

        def in_gap(t):
            if not t: return False
            for gs,ge in gaps:
                if gs<=t<=ge: return True
            return False

        # assign event_ids
        for i,b in enumerate(bundles):
            ectr[0]+=1; b["_eid"]=f"E_{mf24}_{ectr[0]:04d}"

        anchored = {}; assigned_exc = set()

        if has_cgm and not is_flat:
            if is_batch and sig:
                _batch_assign(sig, day_exc, anchored, assigned_exc)
            elif sig:
                _rt_assign(sig, ext_exc, anchored, assigned_exc)

        # stacking
        stacked = {}
        unsig = [(i,b) for i,b in enumerate(bundles) if b["total_CHO"]>CHO_THRESHOLD and i not in anchored]
        for i,b in unsig:
            if not b["reported_time"]: continue
            best_ai, best_gap = None, 1e9
            for ai,(exc,_) in anchored.items():
                ct = bundles[ai].get("_corr", bundles[ai]["reported_time"])
                if not ct: continue
                g = abs((b["reported_time"]-ct).total_seconds()/60)
                if g<best_gap: best_gap=g; best_ai=ai
            if best_ai is not None and best_gap<=STACKING_WINDOW_MIN:
                exc=anchored[best_ai][0]
                if exc["active_start"]<=b["reported_time"]<=exc["active_end"]:
                    stacked[i]=best_ai

        # build results
        for i,b in enumerate(bundles):
            r = {"participant_id":"", "myfood24_id":mf24, "date":d,
                 "meal_label":b["meal_label"], "food_items":b["food_items"],
                 "n_items":b["n_items"]}
            for col in NUTRIENT_COLS:
                r[f"total_{col}"] = round(b.get(f"total_{col}", 0), 4)
            r.update({
                 "reported_time":b["reported_time"], "corrected_time":b["reported_time"],
                 "time_shift_min":0.0, "confidence":"", "match_type":"",
                 "excursion_id":"", "stacked_onto_event_id":"",
                 "excursion_rise_mmol":np.nan, "excursion_peak_mmol":np.nan,
                 "nadir_time":pd.NaT, "peak_time":pd.NaT,
                 "batch_day":is_batch, "time_format":tf,
                 "ampm_resolved_by":b.get("ampm_resolved_by",""),
                 "tz_offset":"", "event_id":b["_eid"]})

            if b["total_CHO"]<=CHO_THRESHOLD:         r["confidence"]="low_cho_no_match"
            elif not has_cgm:                        r["confidence"]="no_cgm_data"
            elif is_flat:                            r["confidence"]="flat_trace"
            elif in_gap(b["reported_time"]):          r["confidence"]="cgm_gap"
            elif i in anchored:
                exc,_ = anchored[i]
                _apply(r, b, exc, is_batch, tf)
            elif i in stacked:
                ai = stacked[i]; exc=anchored[ai][0]
                r["confidence"]="stacked"; r["match_type"]="stacked"
                r["excursion_id"]=exc["excursion_id"]
                r["stacked_onto_event_id"]=bundles[ai]["_eid"]
                r["excursion_rise_mmol"]=exc["rise_mmol"]
                r["excursion_peak_mmol"]=exc["peak_glucose"]
                r["nadir_time"]=exc["nadir_time"]; r["peak_time"]=exc["peak_time"]
            else:
                r["confidence"]="no_match"
            results.append(r)
    return results

def _batch_assign(sig, day_exc, anchored, assigned):
    N,M = len(sig), len(day_exc)
    if M==0 or N==0: return
    sig_s = sorted(sig, key=lambda x: x[1]["reported_time"] or datetime.min)
    if M>=N:
        sel = sorted(sorted(day_exc, key=lambda e:e["rise_mmol"], reverse=True)[:N],
                     key=lambda e:e["est_meal_time"])
    else:
        by_sug = sorted(sig_s, key=lambda x:x[1]["total_TOTSUG"], reverse=True)
        sig_s = sorted(by_sug[:M], key=lambda x:x[1]["reported_time"] or datetime.min)
        sel = sorted(day_exc, key=lambda e:e["est_meal_time"])
    for k in range(min(len(sig_s),len(sel))):
        bi,b = sig_s[k]; exc=sel[k]
        b["_corr"]=exc["est_meal_time"]
        anchored[bi]=(exc,None); assigned.add(exc["excursion_id"])

def _rt_assign(sig, excs, anchored, assigned):
    if not excs or not sig: return
    sug_max = max(b["total_TOTSUG"] for _,b in sig) or 1
    rise_max = max(e["rise_mmol"] for e in excs) or 1
    avail = set(i for i,_ in sig)
    for exc in sorted(excs, key=lambda e:e["est_meal_time"]):
        best_i, best_c = None, 1e9
        for bi,b in sig:
            if bi not in avail or not b["reported_time"]: continue
            sh = abs((b["reported_time"]-exc["est_meal_time"]).total_seconds()/60)
            if sh>MAX_ALLOWABLE_SHIFT_MIN: continue
            tc = W_TIME*sh/60
            ns = b["total_TOTSUG"]/sug_max; nr = exc["rise_mmol"]/rise_max
            sc = W_SUG*abs(ns-nr)
            c = tc+sc
            if c<best_c: best_c=c; best_i=bi
        if best_i is not None:
            for bi,b in sig:
                if bi==best_i:
                    b["_corr"]=exc["est_meal_time"]
                    anchored[bi]=(exc,best_c); avail.discard(bi); assigned.add(exc["excursion_id"])
                    break

def _apply(r, b, exc, is_batch, tf):
    rep = b["reported_time"]; est = exc["est_meal_time"]
    if not rep or not est: r["confidence"]="no_match"; return
    raw = (est-rep).total_seconds()/60
    r["match_type"]="anchor"; r["excursion_id"]=exc["excursion_id"]
    r["excursion_rise_mmol"]=exc["rise_mmol"]; r["excursion_peak_mmol"]=exc["peak_glucose"]
    r["nadir_time"]=exc["nadir_time"]; r["peak_time"]=exc["peak_time"]
    a=abs(raw)
    # Check batch/AM/PM exceptions BEFORE hard rejection —
    # on batch days reported times are meaningless, shifts >180 min are expected
    ampm = b.get("ampm_resolved_by","")
    if a>MAX_ALLOWABLE_SHIFT_MIN:
        if is_batch:
            r["confidence"]="low_batch_override"; r["corrected_time"]=est; r["time_shift_min"]=round(raw,1); return
        if tf=="12h" and ampm=="cgm":
            r["confidence"]="low_ampm_override"; r["corrected_time"]=est; r["time_shift_min"]=round(raw,1); return
        r["confidence"]="rejected_implausible_shift"; return
    if a<30:
        r["confidence"]="high"; r["corrected_time"]=est; r["time_shift_min"]=round(raw,1); return
    if a<90:
        r["confidence"]="medium"; r["corrected_time"]=est; r["time_shift_min"]=round(raw,1); return
    if tf=="12h" and ampm=="cgm":
        r["confidence"]="low_ampm_override"; r["corrected_time"]=est; r["time_shift_min"]=round(raw,1); return
    if is_batch:
        r["confidence"]="low_batch_override"; r["corrected_time"]=est; r["time_shift_min"]=round(raw,1); return
    cl = max(-LOW_CONFIDENCE_CAP_MIN, min(LOW_CONFIDENCE_CAP_MIN, raw))
    r["confidence"]="low_clamped"; r["corrected_time"]=rep+timedelta(minutes=cl); r["time_shift_min"]=round(cl,1)

# ═══════════════════════════════════════════════════════════════════
#  STEP 10
# ═══════════════════════════════════════════════════════════════════
def step10_csv(all_results):
    nutrient_cols = [f"total_{c}" for c in NUTRIENT_COLS]
    cols = (["participant_id","myfood24_id","date","corrected_time","meal_label","food_items","n_items"]
            + nutrient_cols
            + ["reported_time","time_shift_min","confidence","match_type",
               "excursion_id","stacked_onto_event_id","excursion_rise_mmol","excursion_peak_mmol",
               "nadir_time","peak_time","batch_day","time_format","ampm_resolved_by","tz_offset","event_id"])
    df = pd.DataFrame(all_results)
    for c in cols:
        if c not in df.columns: df[c]=""
    df = df[cols]
    out = OUT/"corrected_meal_times_ALL.csv"
    df.to_csv(out, index=False)
    print(f"  corrected_meal_times_ALL.csv  ({len(df)} rows, {len(nutrient_cols)} nutrient columns)")
    return df

def step10_report(all_results, mapping, exc_counts, time_fmts, day_info):
    rows = []
    for mf24,(pid,cgm_p) in mapping.items():
        pr = [r for r in all_results if r["myfood24_id"]==mf24]
        if not pr: continue
        confs = Counter(r["confidence"] for r in pr)
        shifts = [r["time_shift_min"] for r in pr if r["time_shift_min"]!=0]
        rows.append({
            "participant_id":pid, "myfood24_id":mf24,
            "cgm_file_found":cgm_p is not None,
            "time_format":time_fmts.get(mf24,""),
            "tz_offset":pr[0]["tz_offset"] if pr else "",
            "total_meal_events":len(pr),
            "matched_high":confs.get("high",0), "matched_medium":confs.get("medium",0),
            "low_clamped":confs.get("low_clamped",0),
            "low_ampm_override":confs.get("low_ampm_override",0),
            "low_batch_override":confs.get("low_batch_override",0),
            "stacked":confs.get("stacked",0),
            "no_match":confs.get("no_match",0),
            "rejected_shifts":confs.get("rejected_implausible_shift",0),
            "low_cho":confs.get("low_cho_no_match",0),
            "no_cgm":confs.get("no_cgm_data",0),
            "excursions_detected":exc_counts.get(mf24,0),
            "mean_shift_min":round(np.mean(shifts),1) if shifts else 0,
            "median_shift_min":round(np.median(shifts),1) if shifts else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"processing_report.csv", index=False)
    print(f"  processing_report.csv         ({len(df)} rows)")

def step10_plot(all_results, cgm_cache, mapping):
    os.makedirs(OUT/"plots", exist_ok=True)
    pids_done = set()
    for mf24,(pid,cgm_p) in mapping.items():
        if pid in pids_done or cgm_p is None: continue
        pids_done.add(pid)
        pr = [r for r in all_results if r["myfood24_id"]==mf24]
        if not pr: continue
        cgm = cgm_cache.get(mf24)
        if cgm is None: continue

        dates = sorted(set(r["date"] for r in pr))
        cgm_dates = sorted(cgm["date_local"].unique())
        all_dates = sorted(set(dates)|set(cgm_dates))
        nd = len(all_dates)
        if nd==0: continue

        ncols=2; nrows=(nd+1)//2
        fig,axes=plt.subplots(nrows, ncols, figsize=(16, 3*nrows), squeeze=False)
        fig.suptitle(f"{pid} (MF24: {mf24})", fontsize=14, weight="bold")

        for di,d in enumerate(all_dates):
            ax=axes[di//ncols][di%ncols]
            dc = cgm[cgm["date_local"]==d]
            if len(dc)>0:
                ax.plot(dc["ts_local"], dc["glucose"], color="steelblue", lw=0.8)

            day_r = [r for r in pr if r["date"]==d]
            # compute y-range for arrow placement
            _ymin, _ymax = ax.get_ylim() if len(dc)>0 else (4, 10)
            if len(dc)>0:
                _gs = dc["glucose"].dropna()
                if len(_gs)>0:
                    _ymin, _ymax = _gs.min(), _gs.max()
            _arrow_idx = 0
            _label_abbr = {"Breakfast":"B", "Lunch":"L", "Evening dinner":"D",
                           "Snack":"S", "Drink":"Dr"}
            for r in day_r:
                rt = r["reported_time"]; ct = r["corrected_time"]
                ml = r.get("meal_label", "")
                if rt:
                    ax.axvline(rt, color="royalblue", ls="--", alpha=0.5, lw=0.7)
                    abbr = _label_abbr.get(ml, ml[:2] if ml else "")
                    ax.text(mdates.date2num(rt), _ymin - (_ymax - _ymin) * 0.02,
                            abbr, fontsize=5, color="royalblue", fontweight="bold",
                            ha="center", va="top", clip_on=True)
                if ct and rt and ct!=rt:
                    ax.axvline(ct, color="orangered", ls="-", alpha=0.7, lw=0.9)
                    # arrow from reported to corrected time
                    _y_arrow = _ymax - (_ymax - _ymin) * (0.08 + 0.06 * _arrow_idx)
                    shift = r.get("time_shift_min", 0)
                    ax.annotate("",
                                xy=(mdates.date2num(ct), _y_arrow),
                                xytext=(mdates.date2num(rt), _y_arrow),
                                arrowprops=dict(arrowstyle="-|>", color="orangered",
                                                lw=1.2, mutation_scale=10),
                                annotation_clip=True)
                    ax.text(mdates.date2num(rt) + (mdates.date2num(ct) - mdates.date2num(rt)) / 2,
                            _y_arrow + (_ymax - _ymin) * 0.02,
                            f"{shift:+.0f}m", fontsize=5, color="orangered",
                            ha="center", va="bottom", clip_on=True)
                    _arrow_idx += 1
                nt = r.get("nadir_time")
                if pd.notna(nt) and len(dc)>0:
                    diff=(dc["ts_local"]-nt).abs()
                    idx=diff.idxmin()
                    ax.plot(nt, dc.loc[idx,"glucose"], "v", color="green", ms=4, alpha=0.6)

            et = day_info_global.get((mf24,d),"")
            ax.set_title(f"{d}  {'[BATCH]' if et=='batch' else ''}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.grid(True, alpha=0.15)

        for di in range(len(all_dates), nrows*ncols):
            axes[di//ncols][di%ncols].set_visible(False)
        plt.tight_layout()
        fig.savefig(OUT/"plots"/f"{pid}_overview.png", dpi=120)
        plt.close(fig)
    print(f"  {len(pids_done)} participant plots")

def step10_global(all_results):
    df = pd.DataFrame(all_results)
    fig, axes = plt.subplots(2,2, figsize=(14,10))

    # 1. Shift histogram
    shifts = df[df["time_shift_min"]!=0]["time_shift_min"].dropna()
    axes[0,0].hist(shifts, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0,0].set_title("Time shift distribution"); axes[0,0].set_xlabel("Shift (min)")

    # 2. CHO vs rise
    matched = df[df["excursion_rise_mmol"].notna()]
    conf_colors = {"high":"green","medium":"orange","low_clamped":"red",
                   "low_batch_override":"purple","low_ampm_override":"magenta","stacked":"cyan"}
    for c,clr in conf_colors.items():
        sub = matched[matched["confidence"]==c]
        if len(sub): axes[0,1].scatter(sub["total_CHO"], sub["excursion_rise_mmol"],
                                        c=clr, s=8, alpha=0.5, label=c)
    axes[0,1].set_title("CHO vs Excursion Rise"); axes[0,1].set_xlabel("CHO (g)")
    axes[0,1].set_ylabel("Rise (mmol/L)"); axes[0,1].legend(fontsize=6)

    # 3. Confidence pie
    counts = df["confidence"].value_counts()
    axes[1,0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                  textprops={"fontsize":7})
    axes[1,0].set_title("Confidence distribution")

    # 4. Mean shift by meal type
    matched2 = df[(df["time_shift_min"]!=0)&df["meal_label"].notna()]
    if len(matched2)>0:
        ms = matched2.groupby("meal_label")["time_shift_min"].mean()
        axes[1,1].bar(ms.index, ms.values, color="steelblue")
        axes[1,1].set_title("Mean shift by meal type"); axes[1,1].set_ylabel("Shift (min)")
        axes[1,1].tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(OUT/"plots"/"global_summary.png", dpi=120)
    plt.close(fig)
    print(f"  plots/global_summary.png")

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
day_info_global = {}
NUTRIENT_COLS = []

_META_COLS = {
    "Patient Id", "Sex", "Date", "Time consumed at", "Item added at",
    "Meal", "Consumption method", "Food EAN", "Food name",
    "Food category", "Food sub category", "Recipe name",
    "Quantity", "Portion description", "Dry portion multiplier",
    "Total portion size", "Portion unit",
}

def _discover_nutrient_cols(diary):
    """Identify all numeric nutrient columns from the diary DataFrame."""
    cols = []
    for c in diary.columns:
        if c in _META_COLS or c.startswith("Target "):
            continue
        if pd.to_numeric(diary[c], errors="coerce").notna().any():
            cols.append(c)
    return cols

def main():
    global day_info_global, NUTRIENT_COLS
    mapping, diary = step1()
    NUTRIENT_COLS = _discover_nutrient_cols(diary)
    print(f"  {len(NUTRIENT_COLS)} nutrient columns detected")
    time_fmts, day_info = step2(diary, mapping)
    day_info_global = day_info

    all_results = []
    cgm_cache = {}
    exc_counts = {}
    mf24_list = sorted(set(diary["Patient Id"].unique()) & set(mapping.keys()))

    total = len(mf24_list)
    for idx, mf24 in enumerate(mf24_list):
        pid, cgm_path = mapping[mf24]
        tf = time_fmts.get(mf24, "24h")
        diary_sub = diary[diary["Patient Id"]==mf24]
        tz_str = ""

        # Step 3: CGM
        cgm, gaps, cgm_dates = None, [], set()
        if cgm_path:
            cgm, gap_list, tz_str, tz_off = step3(cgm_path, diary_sub)
            gaps = gap_list
            cgm_dates = set(cgm["date_local"].unique())
            cgm_cache[mf24] = cgm

        # Step 4: excursions
        exc_by_date = {}
        if cgm is not None:
            exc_by_date = step4(cgm, mf24)
        n_exc = sum(len(v) for v in exc_by_date.values())
        exc_counts[mf24] = n_exc

        # Step 5: group diary
        bundles_by_date = step5(diary_sub, mf24, None)

        # Step 6: AM/PM
        bundles_by_date = step6(bundles_by_date, exc_by_date, tf)

        # Step 8: match
        results = step8(bundles_by_date, exc_by_date, day_info, tf, mf24, gaps, cgm_dates)

        # Fill tz_offset and participant_id
        for r in results:
            r["tz_offset"] = tz_str
            r["participant_id"] = pid

        n_anc = sum(1 for r in results if r["match_type"]=="anchor")
        n_stk = sum(1 for r in results if r["match_type"]=="stacked")
        n_batch = sum(1 for d in set(r["date"] for r in results) if day_info.get((mf24,d))=="batch")

        print(f"  [{idx+1}/{total}] MF24={mf24} PID={pid:>5s}  CGM  "
              f"exc={n_exc:>3d}  evts={len(results):>3d}  "
              f"anc={n_anc:>3d}  stk={n_stk:>2d}  batch_d={n_batch}")

        all_results.extend(results)

    print(f"\n  Total: {len(all_results)} meal events")

    # Step 10: outputs
    print("\n"+"="*72); print("STEP 10: Produce Outputs"); print("="*72)
    df = step10_csv(all_results)
    step10_report(all_results, mapping, exc_counts, time_fmts, day_info)
    print("  Generating plots...")
    step10_plot(all_results, cgm_cache, mapping)
    step10_global(all_results)

    # Summary
    print("\n"+"="*72); print("  PIPELINE COMPLETE"); print("="*72)
    confs = Counter(r["confidence"] for r in all_results)
    print("  Confidence breakdown:")
    for c,n in confs.most_common():
        print(f"    {c:40s}{n:>5d}  ({100*n/len(all_results):5.1f}%)")
    mt = Counter(r["match_type"] for r in all_results)
    print("  Match types:")
    for m,n in mt.most_common():
        print(f"    {m or 'None':40s}{n:>5d}")
    print(f"\n  Files: corrected_meal_times_ALL.csv, processing_report.csv, "
          f"plots/ ({len(set(r['participant_id'] for r in all_results if mapping.get(r['myfood24_id'],(None,None))[1]))} + global_summary.png)")

if __name__=="__main__":
    print("="*72)
    print("  CGM-First Meal Time Re-Alignment Pipeline  v3")
    print("="*72)
    main()
